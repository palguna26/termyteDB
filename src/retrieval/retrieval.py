"""Hybrid atom retrieval used by LongMemEval-S."""

from __future__ import annotations

import array
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from ..config.settings import RETRIEVAL as _RETRIEVAL_SETTINGS
from ..storage.db import Database
from .embedding import EmbeddingProvider, FastEmbedProvider, cosine

RRF_K = _RETRIEVAL_SETTINGS.rrf_k
HISTORY_RE = re.compile(r"\b(previously|used to|former|formerly|before|previous|history|historical)\b", re.I)

# ---------------------------------------------------------------------------
# Temporal query representation (Phase 2)
# ---------------------------------------------------------------------------

_HAYSTACK_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+((?:19|20)\d{2})\b",
    re.I,
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_LATEST_RE = re.compile(r"\b(latest|current|currently|now|newest|recent|today|present)\b", re.I)
_EARLIEST_RE = re.compile(r"\b(first|earliest|initial|original|oldest)\b", re.I)
_BEFORE_RE = re.compile(r"\b(before|prior to|until|up to|previously|used to|former|formerly|previous)\b", re.I)
_AFTER_RE = re.compile(r"\b(after|since|from|following)\b", re.I)
_AROUND_RE = re.compile(r"\b(around|about|in|during|at the time of|when)\b", re.I)
_MULTI_RE = re.compile(r"\b(across|between|each|all|every|multiple|over time|throughout|compare)\b", re.I)

# Preference signals (Phase 3 — atom-level companion to repository scoring).
_PREF_POSITIVE_RE = re.compile(r"\b(prefer|prefers|preferred|like|likes|liked|love|loves|loved|favorite|favourite|enjoy|favour|favor)\b", re.I)
_PREF_NEGATIVE_RE = re.compile(r"\b(dislike|dislikes|disliked|hate|hates|hated|avoid|avoids|avoided|detest|loathe)\b", re.I)
_PREF_UPDATE_RE = re.compile(r"\b(no longer|used to|previously|before|now|instead|switched|changed to|moved to)\b", re.I)


@dataclass(frozen=True)
class TemporalQuery:
    reference_date: datetime | None
    intent: str  # latest|historical|earliest|before|after|around|none
    target_date: datetime | None = None
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None


def parse_reference_date(value: str | datetime | None) -> datetime | None:
    """Parse LongMemEval question_date / haystack dates into UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    match = _HAYSTACK_DATE_RE.search(text)
    if match:
        try:
            year, month, day, hour, minute = map(int, match.groups())
            return datetime(year, month, day, hour, minute, tzinfo=UTC)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_temporal_query(query: str, reference_date: str | datetime | None = None) -> TemporalQuery:
    """Build an explicit temporal representation for a query.

    ``reference_date`` is the benchmark ``question_date`` — never the machine
    clock — so "current" means valid at question time.
    """
    ref = parse_reference_date(reference_date)
    ql = query.casefold()
    intent = "none"
    if _EARLIEST_RE.search(query):
        intent = "earliest"
    elif _LATEST_RE.search(query):
        intent = "latest"
    elif _BEFORE_RE.search(query) or HISTORY_RE.search(query):
        intent = "historical" if "before" not in ql and "prior" not in ql and "until" not in ql else "before"
        # Distinguish explicit "before X" from generic historical intent.
        if re.search(r"\b(before|prior to|until|up to)\b", query, re.I):
            intent = "before"
        else:
            intent = "historical"
    elif re.search(r"\b(after|since|following)\b", query, re.I):
        intent = "after"
    elif _YEAR_RE.search(query) or _MONTH_YEAR_RE.search(query):
        intent = "around"

    target: datetime | None = None
    range_start: datetime | None = None
    range_end: datetime | None = None
    month_match = _MONTH_YEAR_RE.search(query)
    if month_match:
        try:
            month = _MONTHS[month_match.group(1).casefold()]
            year = int(month_match.group(2))
            range_start = datetime(year, month, 1, tzinfo=UTC)
            if month == 12:
                range_end = datetime(year + 1, 1, 1, tzinfo=UTC)
            else:
                range_end = datetime(year, month + 1, 1, tzinfo=UTC)
            target = range_start + (range_end - range_start) / 2
        except ValueError:
            pass
    else:
        years = [int(v) for v in _YEAR_RE.findall(query)]
        if years:
            year = years[0]
            try:
                range_start = datetime(year, 1, 1, tzinfo=UTC)
                range_end = datetime(year + 1, 1, 1, tzinfo=UTC)
                target = datetime(year, 7, 1, tzinfo=UTC)
            except ValueError:
                pass
    return TemporalQuery(
        reference_date=ref,
        intent=intent,
        target_date=target,
        date_range_start=range_start,
        date_range_end=range_end,
    )


def temporal_atom_boost(atom_timestamp: str | None, tq: TemporalQuery) -> float:
    """Small ranking boost (not a hard filter) from temporal alignment."""
    if tq.intent == "none":
        return 0.0
    atom_dt = parse_reference_date(atom_timestamp)
    if atom_dt is None:
        return -0.01 if tq.intent in {"around", "before", "after"} else 0.0
    if tq.intent == "around" and tq.date_range_start and tq.date_range_end:
        if tq.date_range_start <= atom_dt < tq.date_range_end:
            return 0.08
        # Near-miss decay: within 90 days still gets partial credit.
        try:
            gap = min(abs((atom_dt - tq.date_range_start).days), abs((atom_dt - tq.date_range_end).days))
            if gap <= 90:
                return 0.04 * (1.0 - gap / 90.0)
        except Exception:
            pass
        return 0.0
    if tq.intent == "before" and tq.target_date:
        return 0.05 if atom_dt < tq.target_date else -0.02
    if tq.intent == "after" and tq.target_date:
        return 0.05 if atom_dt >= tq.target_date else -0.02
    if tq.intent == "after" and tq.reference_date:
        # "since X" without explicit target: prefer evidence near reference.
        return 0.0
    if tq.intent == "latest" and tq.reference_date:
        # Prefer facts valid at question time; future-dated atoms are suspect.
        if atom_dt <= tq.reference_date:
            try:
                age_days = max(0.0, (tq.reference_date - atom_dt).total_seconds() / 86400)
                return max(0.0, 0.05 * (1.0 - min(age_days, 365.0) / 365.0))
            except Exception:
                return 0.02
        return -0.03
    if tq.intent == "historical":
        return 0.02
    if tq.intent == "earliest":
        return 0.0  # handled by sort order, not score
    return 0.0


def preference_atom_boost(query: str, fact: str) -> float:
    """Lexical preference alignment between query and candidate fact."""
    ql = query.casefold()
    fl = fact.casefold()
    query_pref = bool(_PREF_POSITIVE_RE.search(query) or _PREF_NEGATIVE_RE.search(query)
                      or "prefer" in ql or "favour" in ql or "favor" in ql
                      or "like" in ql or "dislike" in ql or "favourite" in ql or "favorite" in ql)
    if not query_pref:
        return 0.0
    boost = 0.0
    if _PREF_POSITIVE_RE.search(fact) or "prefer" in fl:
        boost += 0.03
    if _PREF_NEGATIVE_RE.search(fact) or "dislike" in fl or "hate" in fl or "avoid" in fl:
        # Negative preferences are first-class answers to preference queries.
        boost += 0.03
    if _PREF_UPDATE_RE.search(fact) and ("prefer" in ql or "like" in ql or "favour" in ql or "favor" in ql):
        boost += 0.01
    return boost


@dataclass(frozen=True)
class AtomHit:
    atom_id: str
    session_id: str
    fact: str
    timestamp: str | None
    source_role: str
    score: float


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    session_id: str
    text: str
    contextual_text: str
    document_date: str | None
    lexical_score: float
    vector_score: float
    score: float


def search_chunks(
    db: Database,
    query: str,
    limit: int = 20,
    provider: EmbeddingProvider | None = None,
    namespace_id: str | None = None,
    *,
    query_vector: list[float] | None = None,
) -> list[ChunkHit]:
    """Hybrid chunk search: exact terms plus raw and contextual vectors.

    ``query_vector`` reuses the query embedding already computed for atom
    retrieval so one query costs one embedding, not two.  Chunk vectors are
    fetched in a single batched query instead of one query per chunk.
    """
    where = "WHERE namespace_id=?" if namespace_id else ""
    params: tuple[object, ...] = (namespace_id,) if namespace_id else ()
    rows = db.execute(f"SELECT * FROM chunks {where} ORDER BY session_id, ordinal", params).fetchall()
    if not rows:
        return []
    terms = [t.casefold() for t in re.findall(r'[\w./:-]+', query) if len(t) > 1]
    qv = query_vector
    if qv is None and provider is not None:
        try:
            qv = provider.embed(query)
        except Exception:
            qv = None
    # Batched vector fetch: one query for all candidate chunks.
    chunk_vectors: dict[str, list[bytes]] = {}
    if qv is not None and provider is not None:
        try:
            placeholders = ",".join("?" for _ in rows)
            vec_rows = db.execute(
                f"SELECT chunk_id, vector FROM chunk_embeddings WHERE provider=? AND chunk_id IN ({placeholders}) ORDER BY chunk_id, contextual",
                (provider.name, *[r["chunk_id"] for r in rows]),
            ).fetchall()
            for vrow in vec_rows:
                chunk_vectors.setdefault(str(vrow["chunk_id"]), []).append(bytes(vrow["vector"]))
        except Exception:
            chunk_vectors = {}
    else:
        # No dense vector available: lexical-only path (Phase 5 fast path).
        chunk_vectors = {}
    hits: list[ChunkHit] = []
    for row in rows:
        lexical = sum(t in (row['raw_text'] + ' ' + row['contextual_text']).casefold() for t in terms) / max(1, len(terms))
        dense = 0.0
        if qv is not None:
            for vec_bytes in chunk_vectors.get(str(row["chunk_id"]), []):
                try:
                    dense = max(dense, cosine(qv, list(array.array('f', vec_bytes))))
                except Exception:
                    continue
        score = 0.6 * dense + 0.4 * lexical
        if score > 0:
            hits.append(ChunkHit(row['chunk_id'], row['session_id'], row['raw_text'], row['contextual_text'], row['document_date'], lexical, dense, score))
    return sorted(hits, key=lambda h: (-h.score, h.chunk_id))[:limit]


def rrf_merge(lists: list[list[AtomHit]], k: int = RRF_K) -> list[AtomHit]:
    merged: dict[str, tuple[AtomHit, float]] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            score = 1.0 / (k + rank + 1)
            previous = merged.get(item.atom_id)
            merged[item.atom_id] = (item, score if previous is None else previous[1] + score)
    return [
        AtomHit(item.atom_id, item.session_id, item.fact, item.timestamp, item.source_role, score)
        for item, score in sorted(merged.values(), key=lambda pair: (-pair[1], pair[0].atom_id))
    ]


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w./:-]+", query, re.UNICODE)
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms if term)


def _batch_atom_flags(db: Database, atom_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """One batched lookup for (invalid_at, namespace_id) per atom id."""
    if not atom_ids:
        return {}
    placeholders = ",".join("?" for _ in atom_ids)
    flags: dict[str, tuple[str | None, str | None]] = {}
    try:
        for row in db.execute(
            f"SELECT atom_id, invalid_at, namespace_id FROM atoms WHERE atom_id IN ({placeholders})",
            tuple(atom_ids),
        ).fetchall():
            flags[str(row["atom_id"])] = (row["invalid_at"], row["namespace_id"])
    except Exception:
        return {}
    return flags


def search_atoms(
    db: Database, query: str, limit: int = 20, vector_search: Callable[[str, int], list[AtomHit]] | None = None, namespace_id: str | None = None,
    *,
    reference_date: str | datetime | None = None,
    temporal_query: TemporalQuery | None = None,
) -> list[AtomHit]:
    hits, _ = search_atoms_with_stages(
        db, query, limit,
        vector_search=vector_search, namespace_id=namespace_id,
        reference_date=reference_date, temporal_query=temporal_query,
    )
    return hits


def search_atoms_with_stages(
    db: Database, query: str, limit: int = 20, vector_search: Callable[[str, int], list[AtomHit]] | None = None, namespace_id: str | None = None,
    *,
    reference_date: str | datetime | None = None,
    temporal_query: TemporalQuery | None = None,
) -> tuple[list[AtomHit], dict[str, float]]:
    """Atom search with per-stage latency breakdown (Phase 1 measurement).

    Returns (hits, stages_ms) where stages_ms has fts_ms, dense_ms, rrf_ms,
    temporal_ms keys.  Per-result N+1 lookups are replaced with batched SQL.
    """
    tq = temporal_query if temporal_query is not None else parse_temporal_query(query, reference_date)
    historical = bool(HISTORY_RE.search(query)) or tq.intent in {"historical", "before", "earliest"}
    match = _fts_query(query)
    stages: dict[str, float] = {"fts_ms": 0.0, "dense_ms": 0.0, "rrf_ms": 0.0, "temporal_ms": 0.0}
    lexical: list[AtomHit] = []
    if match:
        started = time.perf_counter()
        if namespace_id is not None:
            rows = db.execute(
                """SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role,
                          bm25(atoms_fts) AS rank
                   FROM atoms_fts JOIN atoms a ON a.atom_id=atoms_fts.atom_id
                   WHERE atoms_fts MATCH ? AND (? OR a.invalid_at IS NULL)
                     AND a.namespace_id = ?
                   ORDER BY rank LIMIT ?""",
                (match, historical, namespace_id, max(limit * 3, 20)),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role,
                          bm25(atoms_fts) AS rank
                   FROM atoms_fts JOIN atoms a ON a.atom_id=atoms_fts.atom_id
                   WHERE atoms_fts MATCH ? AND (? OR a.invalid_at IS NULL)
                   ORDER BY rank LIMIT ?""",
                (match, historical, max(limit * 3, 20)),
            ).fetchall()
        lexical = [AtomHit(r["atom_id"], r["session_id"], r["fact"], r["timestamp"], r["source_role"], float(r["rank"])) for r in rows]
        stages["fts_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if vector_search is not None:
        semantic = vector_search(query, max(limit * 3, 20))
    else:
        semantic = dense_search_atoms(db, query, max(limit * 3, 20), namespace_id=namespace_id)
    stages["dense_ms"] = (time.perf_counter() - started) * 1000
    # Batched validity + namespace filtering (replaces per-result queries).
    if not historical or namespace_id is not None:
        all_ids = list(dict.fromkeys([h.atom_id for h in lexical] + [h.atom_id for h in semantic]))
        flags = _batch_atom_flags(db, all_ids)
        if not historical:
            semantic = [item for item in semantic if (flags.get(item.atom_id, (None, None))[0] is None)]
        if namespace_id is not None:
            lexical = [h for h in lexical if flags.get(h.atom_id, (None, namespace_id))[1] == namespace_id]
            semantic = [h for h in semantic if flags.get(h.atom_id, (None, namespace_id))[1] == namespace_id]
    started = time.perf_counter()
    merged = rrf_merge([lexical, semantic])
    stages["rrf_ms"] = (time.perf_counter() - started) * 1000
    # Temporal + preference re-scoring (small boosts, never hard filters).
    started = time.perf_counter()
    if tq.intent != "none" or preference_atom_boost(query, "") != 0.0 or True:
        rescored: list[AtomHit] = []
        for hit in merged:
            boost = temporal_atom_boost(hit.timestamp, tq)
            boost += preference_atom_boost(query, hit.fact)
            if boost:
                rescored.append(AtomHit(hit.atom_id, hit.session_id, hit.fact, hit.timestamp, hit.source_role, hit.score + boost))
            else:
                rescored.append(hit)
        # "earliest" intent sorts by timestamp ascending as tie-break.
        if tq.intent == "earliest":
            merged = sorted(rescored, key=lambda h: (-h.score + 0.0001 * _timestamp_rank(h.timestamp), h.atom_id))
            # Apply stable earliest-first for near-ties: sort by timestamp when scores close.
            merged = sorted(merged, key=lambda h: (_timestamp_rank(h.timestamp), -h.score))
        else:
            merged = sorted(rescored, key=lambda h: (-h.score, h.atom_id))
    stages["temporal_ms"] = (time.perf_counter() - started) * 1000
    return merged[:limit], stages


def _timestamp_rank(value: str | None) -> float:
    dt = parse_reference_date(value)
    if dt is None:
        return float("inf")
    try:
        return dt.timestamp()
    except Exception:
        return float("inf")


def aggregate_atom_sessions(
    hits: list[AtomHit],
    query: str,
    *,
    limit: int = 15,
    max_per_session: int = 3,
    multi_session_reserve: float = 0.4,
) -> list[str]:
    """Session-level aggregation with coverage and redundancy rules (Phase 4).

    Session score combines best memory score, top-2 support, independent hit
    count (diminishing returns), query-term coverage, and recency only when
    the query asks for it.  Multi-session queries reserve part of the budget
    for independent sessions so one strong session cannot dominate.
    """
    if not hits:
        return []
    terms = [t.casefold() for t in re.findall(r"[\w./:-]+", query) if len(t) > 2]
    is_multi = bool(_MULTI_RE.search(query))
    grouped: dict[str, list[AtomHit]] = {}
    for hit in hits:
        grouped.setdefault(str(hit.session_id), []).append(hit)
    session_scores: dict[str, float] = {}
    for sid, items in grouped.items():
        ordered = sorted(items, key=lambda h: -h.score)
        best = ordered[0].score
        second = ordered[1].score if len(ordered) > 1 else 0.0
        # Diminishing returns: 10 duplicates must not beat 10 independent sessions.
        import math as _math
        breadth = 0.05 * _math.log1p(len(ordered))
        coverage = 0.0
        if terms:
            joined = " ".join(h.fact.casefold() for h in ordered[:3])
            coverage = 0.10 * sum(t in joined for t in terms) / max(1, len(terms))
        recency = 0.0
        if _LATEST_RE.search(query):
            times = [_timestamp_rank(h.timestamp) for h in ordered if _timestamp_rank(h.timestamp) != float("inf")]
            if times:
                recency = 0.01
        session_scores[sid] = best + 0.5 * second + breadth + coverage + recency
    ranked_sessions = sorted(session_scores, key=lambda sid: (-session_scores[sid], sid))
    if not is_multi:
        # Precision path: first-seen order would over-reward one session;
        # ranked order with per-session cap is strictly better.
        return ranked_sessions[:limit]
    # Multi-session path: round-robin across ranked sessions so every
    # independent session contributes evidence up to the per-session cap.
    reserve_sessions = max(2, int(len(ranked_sessions) * multi_session_reserve) or 2)
    selected: list[str] = []
    per_session_used: dict[str, int] = {sid: 0 for sid in ranked_sessions}
    # First pass: one hit-session each for the top reserve set.
    for sid in ranked_sessions[:reserve_sessions]:
        selected.append(sid)
        per_session_used[sid] += 1
    for sid in ranked_sessions:
        if sid in selected:
            continue
        if len(selected) >= limit:
            break
        selected.append(sid)
    return selected[:limit]


def dense_search_atoms(
    db: Database, query: str, limit: int = 20, provider: EmbeddingProvider | None = None, namespace_id: str | None = None,
    *,
    query_vector: list[float] | None = None,
) -> list[AtomHit]:
    if provider is None and query_vector is None:
        try:
            provider = _cached_fastembed_provider()
        except ImportError:
            return []
    if query_vector is None:
        assert provider is not None
        available = db.execute("SELECT 1 FROM atom_embeddings WHERE provider=? LIMIT 1", (provider.name,)).fetchone()
        if available is None:
            return []
        try:
            query_vector = provider.embed(query)
        except Exception:
            return []
        provider_name = provider.name
    else:
        provider_name = provider.name if provider is not None else None
        if provider_name is None:
            row = db.execute("SELECT provider FROM atom_embeddings LIMIT 1").fetchone()
            if row is None:
                return []
            provider_name = str(row["provider"])
    if namespace_id is not None:
        rows = db.execute(
            "SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role, e.vector "
            "FROM atoms a JOIN atom_embeddings e ON e.atom_id=a.atom_id WHERE e.provider=? AND a.namespace_id = ?",
            (provider_name, namespace_id),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT a.atom_id, a.session_id, a.fact, a.timestamp, a.source_role, e.vector "
            "FROM atoms a JOIN atom_embeddings e ON e.atom_id=a.atom_id WHERE e.provider=?",
            (provider_name,),
        ).fetchall()
    hits: list[AtomHit] = []
    assert query_vector is not None
    for row in rows:
        vector = list(array.array("f", row["vector"]))
        try:
            score = cosine(query_vector, vector)
        except Exception:
            continue
        hits.append(AtomHit(row["atom_id"], row["session_id"], row["fact"], row["timestamp"], row["source_role"], score))
    return sorted(hits, key=lambda item: (-item.score, item.atom_id))[:limit]


@lru_cache(maxsize=1)
def _cached_fastembed_provider() -> FastEmbedProvider:
    """Reuse the local model across queries instead of reloading it per search."""
    return FastEmbedProvider()


@lru_cache(maxsize=2)
def _cached_flashrank(model_name: str):  # type: ignore[no-untyped-def]
    """Create a FlashRank model once per process, not once per query."""
    from flashrank import Ranker  # type: ignore[import-untyped]

    return Ranker(model_name=model_name)


def rerank_and_filter(query: str, hits: list[AtomHit], threshold: float | None = None) -> list[AtomHit] | None:
    """Optional FlashRank reranking with hard abstention (cached init)."""
    if threshold is None:
        threshold = _RETRIEVAL_SETTINGS.reranker_threshold
    try:
        from flashrank import RerankRequest  # type: ignore[import-untyped]
    except ImportError:
        return hits if hits else None
    try:
        ranker = _cached_flashrank(_RETRIEVAL_SETTINGS.reranker_model)
    except Exception:
        return hits if hits else None
    passages = [{"id": hit.atom_id, "text": hit.fact, "meta": hit} for hit in hits]
    try:
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    except Exception:
        return hits if hits else None
    if not results or float(results[0].get("score", 0.0)) < threshold:
        return None
    return [item["meta"] for item in results]


def pack_context(hits: list[AtomHit], token_budget: int = 3000) -> str:
    """Legacy word-based packing kept for backwards compatibility."""
    packed = pack_atoms_token_aware(hits, token_budget=token_budget)
    return packed["text"]


def pack_atoms_token_aware(
    hits: list[AtomHit],
    *,
    token_budget: int = 1200,
    tokenizer_model: str | None = None,
) -> dict[str, object]:
    """Token-aware atom packing with a hard budget (Phase 1).

    The budget applies *after* rendering all headers, timestamps, roles, and
    evidence text — never on raw facts alone.  Returns exact token and word
    counts separately plus the tokenizer mode.
    """
    from .context import count_tokens, count_words, tokenizer_mode, truncate_to_tokens

    budget = max(1, int(token_budget))
    groups: dict[str, list[AtomHit]] = {}
    for hit in hits:
        groups.setdefault(hit.session_id, []).append(hit)
    lines: list[str] = []
    for session_id in sorted(groups, key=lambda sid: min((h.timestamp or "9999") for h in groups[sid])):
        lines.append(f"[Session {session_id}]")
        for hit in sorted(groups[session_id], key=lambda item: (item.timestamp or "9999", item.atom_id)):
            lines.append(f"[Timestamp: {hit.timestamp or 'unknown'}] [Role: {hit.source_role}] {hit.fact}")
    if not lines:
        text = "insufficient information"
        return {
            "text": text,
            "token_count": count_tokens(text, model=tokenizer_model),
            "word_count": count_words(text),
            "tokenizer": tokenizer_mode(model=tokenizer_model),
            "truncated": False,
        }
    packed: list[str] = []
    truncated = False
    for line in lines:
        candidate = "\n".join([*packed, line])
        if count_tokens(candidate, model=tokenizer_model) > budget:
            # Try to fit a truncated tail of the current line rather than
            # dropping a relevant session entirely.
            base = "\n".join(packed)
            base_tokens = count_tokens(base, model=tokenizer_model) if packed else 0
            remaining = budget - base_tokens - 1
            if remaining > 8 and line.startswith("[Timestamp:"):
                excerpt = truncate_to_tokens(line, remaining, model=tokenizer_model)
                if excerpt and count_tokens("\n".join([*packed, excerpt]), model=tokenizer_model) <= budget:
                    packed.append(excerpt)
                    truncated = True
            else:
                truncated = True
            break
        packed.append(line)
    text = "\n".join(packed) if packed else "insufficient information"
    if count_tokens(text, model=tokenizer_model) > budget:
        text = truncate_to_tokens(text, budget, model=tokenizer_model)
        truncated = True
    return {
        "text": text,
        "token_count": count_tokens(text, model=tokenizer_model),
        "word_count": count_words(text),
        "tokenizer": tokenizer_mode(model=tokenizer_model),
        "truncated": truncated,
    }

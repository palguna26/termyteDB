"""LongMemEval-S benchmark for TermyteDB.

Canonical harness supporting two distinct benchmark pipelines:

* retrieval-only (``retrieval`` / ``retrieval-only``) — lossless episodic retrieval
  isolation benchmark: verbatim turn-level atoms -> FTS5 + dense hybrid (RRF) ->
  FlashRank rerank -> session aggregation -> bounded context packing. This is the
  existing upper-bound benchmark; it shortcuts memory formation and measures
  indexing + retrieval only.

* end-to-end (``end-to-end``) — true production memory pipeline:
  LongMemEval conversation/session history -> EventInput ingestion ->
  processing job -> extraction (rule / OpenRouter / fake) -> evidence validation
  -> reconciliation/versioning -> embeddings/indexing -> hybrid retrieval ->
  context packing -> retrieval metrics.

  No ground-truth question, answer, answer aliases, relevance annotations,
  evidence identifiers, or category hints leak into memory formation.
  The question appears only after processing completes.

General methodology notes are in docs/benchmarks.md; additional pipeline
details for end-to-end are in the docstring of ``ingest_and_process_e2e``
and ``evaluate_sample_e2e``.

Modes:
  retrieval / retrieval-only  Zero-cost session-level retrieval (atoms)
  end-to-end                 Production pipeline via TermyteDB events/memories
  judged                     retrieval-only + OpenRouter answer generation/judging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from termytedb.evaluation.longmemeval_extraction import L1Atom, index_atom_embeddings, insert_atoms  # noqa: E402
from termytedb.retrieval.embedding import FastEmbedProvider  # noqa: E402
from termytedb.retrieval.retrieval import AtomHit, dense_search_atoms, pack_context, search_atoms  # noqa: E402
from termytedb.storage.db import Database  # noqa: E402

DEFAULT_DATA_PATH = ROOT / "benchmarks" / "longmemeval" / "longmemeval_s_cleaned.json"
CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
]

_ranker = None
_ranker_lock = threading.Lock()
_embedder: GuardedEmbedder | None = None
_product_embedder: GuardedEmbedder | None = None
_product_embedder_key: tuple[str, str | None, int | None] | None = None
_product_embedder_lock = threading.Lock()
_RUN_MANIFEST_NAME = "longmemeval_run_manifest.json"


class GuardedEmbedder:
    """Serializes ONNX inference so concurrent workers cannot grow competing arenas."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, value: str) -> list[float]:
        with self._lock:
            return self._inner.embed(value)

    def embed_many(self, values: list[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_many(values)


def shared_ranker(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    global _ranker
    if _ranker is None:
        with _ranker_lock:
            if _ranker is None:
                from flashrank import Ranker  # type: ignore[import-untyped]

                _ranker = Ranker(model_name=model_name)
    return _ranker


def shared_embedder() -> GuardedEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = GuardedEmbedder(FastEmbedProvider(batch_size=16, threads=1))
    return _embedder


def shared_product_embedder(args: argparse.Namespace) -> GuardedEmbedder:
    global _product_embedder, _product_embedder_key
    provider_name = getattr(args, "embedding_provider", None) or os.environ.get("TERMYTEDB_EMBEDDING_PROVIDER", "local")
    model_name = getattr(args, "embedding_model", None) or os.environ.get("TERMYTEDB_EMBEDDING_MODEL")
    dimensions = getattr(args, "embedding_dimensions", None)
    if dimensions is None and provider_name == "openrouter":
        dimensions = int(os.environ.get("TERMYTEDB_EMBEDDING_DIMENSIONS", "1536"))
    key = (provider_name, model_name, dimensions)
    if _product_embedder is None or _product_embedder_key != key:
        with _product_embedder_lock:
            if _product_embedder is None or _product_embedder_key != key:
                if provider_name == "openrouter":
                    from termytedb.retrieval.embedding import OpenAICompatibleEmbeddingProvider  # noqa: E402

                    _product_embedder = GuardedEmbedder(
                        OpenAICompatibleEmbeddingProvider(model_name, dimensions=dimensions)
                    )
                else:
                    _product_embedder = shared_embedder()
                _product_embedder_key = key
    return _product_embedder


_rerank_mutex = threading.Lock()


def rerank_hits(query: str, hits: list[AtomHit], threshold: float, *, max_candidates: int = 30, max_chars: int = 600) -> list[AtomHit] | None:
    """Shared-instance FlashRank rerank with hard abstention."""
    from flashrank import RerankRequest  # type: ignore[import-untyped]

    ranker = shared_ranker()
    candidates = hits[:max_candidates]
    tail = hits[max_candidates:]
    passages = [{"id": hit.atom_id, "text": hit.fact[:max_chars]} for hit in candidates]
    with _rerank_mutex:
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    scored = [(entry["id"], float(entry.get("score", 0.0))) for entry in results]
    by_id = {hit.atom_id: hit for hit in candidates}
    ordered = [by_id[atom_id] for atom_id, _ in scored if atom_id in by_id]
    if not ordered or scored[0][1] < threshold:
        return None
    return ordered + tail


def rerank_memory_hits(query: str, hits: list[Any], threshold: float, *, max_candidates: int = 30, max_chars: int = 600) -> list[Any] | None:
    """FlashRank rerank for TermyteDB SearchResult objects."""
    from flashrank import RerankRequest  # type: ignore[import-untyped]

    ranker = shared_ranker()
    candidates = hits[:max_candidates]
    tail = hits[max_candidates:]
    passages = [{"id": str(h.memory_version_id), "text": h.statement[:max_chars]} for h in candidates]
    with _rerank_mutex:
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    scored = [(entry["id"], float(entry.get("score", 0.0))) for entry in results]
    by_id = {str(h.memory_version_id): h for h in candidates}
    ordered = [by_id[mid] for mid, _ in scored if mid in by_id]
    if not ordered or scored[0][1] < threshold:
        return None
    return ordered + tail


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    question_type: str
    answer: str
    answer_session_ids: frozenset[str]
    unanswerable: bool
    sessions: tuple[tuple[str, str, tuple[dict[str, str], ...]], ...]
    raw_words: int


def normalize_samples(raw: Any) -> list[Sample]:
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("samples", []))
    samples: list[Sample] = []
    for item in items:
        sessions: list[tuple[str, str, tuple[dict[str, str], ...]]] = []
        raw_words = 0
        ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])
        for index, messages in enumerate(item.get("haystack_sessions", [])):
            session_id = str(ids[index]) if index < len(ids) else f"session-{index}"
            date = str(dates[index]) if index < len(dates) else ""
            turns = tuple(
                {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
                for message in messages
                if isinstance(message, dict)
            )
            raw_words += sum(len(turn["content"].split()) for turn in turns)
            sessions.append((session_id, date, turns))
        samples.append(
            Sample(
                question_id=str(item.get("question_id", len(samples))),
                question=str(item.get("question", "")),
                question_type=str(item.get("question_type", "unknown")),
                answer=str(item.get("answer", "")),
                answer_session_ids=frozenset(str(value) for value in item.get("answer_session_ids", [])),
                unanswerable=bool(item.get("unanswerable", False)),
                sessions=tuple(sessions),
                raw_words=raw_words,
            )
        )
    return samples


MAX_ATOM_CHARS = 1500


def verbatim_atoms(sample: Sample, *, namespace_id: str | None = None) -> list[L1Atom]:
    """One atom per message: lossless episodic encoding at zero API cost."""
    atoms: list[L1Atom] = []
    for session_id, date, turns in sample.sessions:
        for turn in turns:
            fact = turn["content"][:MAX_ATOM_CHARS]
            atoms.append(
                L1Atom(
                    atom_id=str(uuid4()),
                    session_id=session_id,
                    fact=fact,
                    timestamp=date or None,
                    source_role=turn["role"],
                    namespace_id=namespace_id,
                )
            )
    return atoms


_single_db_lock = threading.Lock()


def _ensure_namespace(db: Database, namespace_id: str) -> None:
    with db.connection:
        db.execute(
            "INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES (?, ?, datetime('now'))",
            (namespace_id, "benchmark"),
        )


def ingest_sample(
    work_dir: Path, sample: Sample, *, skip_embeddings: bool = False, single_db: bool = False
) -> Path:
    if single_db:
        database_path = work_dir / "single.sqlite"
        # Serialize writes to the single file to avoid WAL contention
        with _single_db_lock:
            db = Database(database_path)
            try:
                _ensure_namespace(db, sample.question_id)
                existing = db.execute(
                    "SELECT COUNT(*) FROM atoms WHERE namespace_id = ?", (sample.question_id,)
                ).fetchone()[0]
                if existing == 0:
                    insert_atoms(db, verbatim_atoms(sample, namespace_id=sample.question_id))
                    if not skip_embeddings:
                        index_atom_embeddings(db, shared_embedder(), batch_size=64)
                elif not skip_embeddings:
                    missing = db.execute(
                        """SELECT COUNT(*) FROM atoms a
                           LEFT JOIN atom_embeddings e ON e.atom_id=a.atom_id AND e.provider=?
                           WHERE e.atom_id IS NULL AND a.namespace_id = ?""",
                        (shared_embedder().name, sample.question_id),
                    ).fetchone()[0]
                    if missing:
                        index_atom_embeddings(db, shared_embedder(), batch_size=64)
            finally:
                db.close()
        return database_path

    database_path = work_dir / f"{sample.question_id}.sqlite"
    db = Database(database_path)
    try:
        existing = db.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
        if existing == 0:
            insert_atoms(db, verbatim_atoms(sample))
            if not skip_embeddings:
                index_atom_embeddings(db, shared_embedder(), batch_size=64)
        elif not skip_embeddings:
            missing = db.execute(
                """SELECT COUNT(*) FROM atoms a
                   LEFT JOIN atom_embeddings e ON e.atom_id=a.atom_id AND e.provider=?
                   WHERE e.atom_id IS NULL""",
                (shared_embedder().name,),
            ).fetchone()[0]
            if missing:
                index_atom_embeddings(db, shared_embedder(), batch_size=64)
    finally:
        db.close()
    return database_path


def shared_dense(db: Database, query: str, limit: int, namespace_id: str | None = None) -> list[AtomHit]:
    return dense_search_atoms(db, query, limit, provider=shared_embedder(), namespace_id=namespace_id)


# ---------------------------------------------------------------------------
# End-to-end helpers
# ---------------------------------------------------------------------------

_HAYSTACK_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def _parse_haystack_date(value: str) -> datetime | None:
    """Parse haystack date like '2023/05/20 (Sat) 02:21' into UTC datetime."""
    if not value:
        return None
    match = _HAYSTACK_DATE_RE.search(value)
    if not match:
        return None
    try:
        year, month, day, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute, tzinfo=UTC)
    except ValueError:
        return None


def build_event_inputs(sample: Sample) -> list[dict[str, Any]]:
    """Convert LongMemEval sessions/turns into production EventInput dicts.

    Important leakage boundary: only haystack_sessions, haystack_session_ids,
    and haystack_dates are used. Question, answer, answer_session_ids and
    relevance annotations are never included.
    """
    events: list[dict[str, Any]] = []
    for session_index, (session_id, date_str, turns) in enumerate(sample.sessions):
        base_time = _parse_haystack_date(date_str)
        for turn_index, turn in enumerate(turns):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # Deterministic idempotency key per turn, stable across runs.
            # Include session_index to avoid collisions when haystack contains duplicate session_ids.
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            idempotency_key = f"longmemeval:{sample.question_id}:{session_index}:{session_id}:{turn_index}:{content_hash}"
            occurred = base_time + timedelta(seconds=turn_index * 60) if base_time else None
            # Payload uses explicit messages field so payload_text preserves speaker
            payload: dict[str, Any] = {
                "messages": [{"role": role, "content": content}],
                "session_id": session_id,
            }
            if date_str:
                payload["haystack_date"] = date_str
            event_dict: dict[str, Any] = {
                "namespace_id": sample.question_id,
                "idempotency_key": idempotency_key,
                "type": "conversation",
                "payload": payload,
                "stream_id": session_id,
                "session_id": session_id,
            }
            if role in ("user", "assistant"):
                event_dict["actor_id"] = role
            if occurred is not None:
                event_dict["occurred_at"] = occurred.isoformat()
            events.append(event_dict)
    return events


def build_provider(args: argparse.Namespace):
    """Factory for extraction providers."""
    name = getattr(args, "extraction", "openrouter")
    if name == "openrouter":
        from termytedb.memory.provider import OpenRouterExtractionProvider  # noqa: E402

        model = getattr(args, "extraction_model", None) or os.environ.get("TERMYTEDB_EXTRACTION_MODEL") or "openrouter/free"
        return OpenRouterExtractionProvider(model=model)
    if name == "rule":
        return None
    if name == "fake":
        from termytedb.memory.provider import FakeExtractionProvider  # noqa: E402

        # For unit tests a deterministic fake provider; for benchmark treat as rule-mode
        return FakeExtractionProvider()
    if name == "http":
        from termytedb.memory.provider import HttpExtractionProvider  # noqa: E402

        return HttpExtractionProvider()
    raise ValueError(f"unknown extraction provider: {name}")


def _e2e_database_path(work_dir: Path, sample: Sample, single_db: bool) -> Path:
    if single_db:
        return work_dir / "e2e_single.sqlite"
    return work_dir / f"e2e_{sample.question_id}.sqlite"


def _run_manifest_path(work_dir: Path) -> Path:
    return work_dir / _RUN_MANIFEST_NAME


def _run_manifest(args: argparse.Namespace, data_path: Path, dataset_sha256: str, canonical_mode: str) -> dict[str, Any]:
    return {
        "mode": canonical_mode,
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "extraction": getattr(args, "extraction", None),
        "extraction_model": getattr(args, "extraction_model", None),
        "embedding_provider": getattr(args, "embedding_provider", None),
        "embedding_model": getattr(args, "embedding_model", None),
        "embedding_dimensions": getattr(args, "embedding_dimensions", None),
        "single_db": bool(getattr(args, "single_db", False)),
        "no_dense": bool(getattr(args, "no_dense", False)),
        "no_rerank": bool(getattr(args, "no_rerank", False)),
        "recall_k": int(getattr(args, "recall_k", 15)),
        "token_budget": int(getattr(args, "token_budget", 1500)),
        "pack_atoms": int(getattr(args, "pack_atoms", 40)),
        "abstain_threshold": float(getattr(args, "abstain_threshold", 0.25)),
    }


def ingest_and_process_e2e(
    work_dir: Path, sample: Sample, args: argparse.Namespace
) -> tuple[Path, dict[str, Any]]:
    """Ingest LongMemEval history through production pipeline and process.

    Returns (database_path, diagnostics).
    """
    # Lazy imports to avoid heavy deps at module import time
    from termytedb.api.schemas import EventInput  # noqa: E402
    from termytedb.runtime.engine import TermyteDB  # noqa: E402

    single_db = bool(getattr(args, "single_db", False))
    database_path = _e2e_database_path(work_dir, sample, single_db)
    skip_process = False

    # Check existing state for resume/idempotency in single-db mode
    # Per-question isolation: namespace = question_id
    namespace_id = sample.question_id
    provider = build_provider(args)
    # Build embedding provider: shared guarded provider for reuse
    embedding_provider = shared_product_embedder(args)

    # Use file lock for single-db concurrent access
    lock = _single_db_lock if single_db else threading.Lock()
    # Serialize entire ingest+process for single_db to avoid WAL races
    with lock:
        engine = TermyteDB(database_path, extraction_provider=provider, embedding_provider=embedding_provider)  # type: ignore[arg-type]
        try:
            events_raw = build_event_inputs(sample)
            # Filter to EventInput for validation
            events_input = [EventInput.model_validate(e) for e in events_raw]

            ingest_started = time.perf_counter()
            events_ingested = 0
            events_duplicate = 0
            receipts = []
            for ev in events_input:
                receipt = engine.ingest(ev)
                receipts.append(receipt)
                if receipt.duplicate:
                    events_duplicate += 1
                else:
                    events_ingested += 1
            ingest_latency_ms = (time.perf_counter() - ingest_started) * 1000

            # Drain processing jobs
            process_started = time.perf_counter()
            total_processed = total_failed = total_dead = total_accepted = total_rejected = 0
            batch_size = int(getattr(args, "processing_batch_size", 100) or 100)
            lease_seconds = int(getattr(args, "processing_lease_seconds", 180) or 180)
            timeout_seconds = float(getattr(args, "processing_timeout", 30.0) or 30.0)
            # Loop until no pending jobs remain (or timeout per call)
            for _ in range(50):  # safety cap: 50 batches per sample
                resp = engine.process(namespace_id, limit=batch_size, lease_seconds=lease_seconds)
                # Also try process_with_timeout? Use process
                total_processed += resp.processed
                total_failed += resp.failed
                total_dead += resp.dead_lettered
                total_accepted += resp.accepted
                total_rejected += resp.rejected
                if resp.processed == 0 and resp.failed == 0:
                    break
                # Check if jobs remain pending
                metrics = engine.metrics(namespace_id)
                if metrics.get("jobs_pending", 0) == 0 and metrics.get("jobs_processing", 0) == 0:
                    break
            process_latency_ms = (time.perf_counter() - process_started) * 1000

            # Collect post-process diagnostics from repository
            metrics_final = engine.metrics(namespace_id)
            runs = engine.extraction_runs(namespace_id, limit=1000)
            decisions = engine.extraction_decisions(namespace_id, limit=1000)
            memories = engine.memories(namespace_id, limit=1000)
            # Rejection reasons
            rejection_counter = Counter(d.get("rejection_reason") for d in decisions if d.get("validation_status") == "rejected" and d.get("rejection_reason"))
            # If dense disabled, optionally clear embeddings so retrieval becomes lexical-only
            if getattr(args, "no_dense", False):
                try:
                    engine.database.execute(
                        "DELETE FROM memory_embeddings WHERE namespace_id=?", (namespace_id,)
                    )
                    engine.database.connection.commit()
                except Exception:
                    pass

            diagnostics: dict[str, Any] = {
                "events_ingested": events_ingested,
                "events_duplicate": events_duplicate,
                "events_total": len(events_input),
                "ingest_latency_ms": round(ingest_latency_ms, 2),
                "processing_jobs_completed": total_processed,
                "processing_jobs_failed": total_failed,
                "processing_jobs_dead": total_dead,
                "candidates_extracted": total_accepted + total_rejected,
                "candidates_accepted": total_accepted,
                "candidates_rejected": total_rejected,
                "memories_created": len(memories),
                "memory_versions_created": int(metrics_final.get("memory_versions", 0)),
                "average_memories_per_sample": len(memories),
                "extraction_latency_ms": round(process_latency_ms, 2),
                "processing_latency_ms": round(process_latency_ms, 2),
                "metrics": metrics_final,
                "rejection_reasons": dict(rejection_counter),
                "runs_count": len(runs),
                "decisions_count": len(decisions),
            }
        finally:
            engine.close()
    return database_path, diagnostics


def retrieve_e2e_session_ranking(database_path: Path, sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    """Retrieve after end-to-end processing using production memories."""
    from termytedb.runtime.engine import TermyteDB  # noqa: E402

    ns = sample.question_id
    provider = build_provider(args)
    engine = TermyteDB(database_path, extraction_provider=provider, embedding_provider=shared_product_embedder(args))  # type: ignore[arg-type]
    started = time.perf_counter()
    try:
        limit = max(args.recall_k * 10, 50)
        # Repository.search hybrid retrieval
        search_results = engine.search(ns, sample.question, limit=limit)
        # Rerank if enabled
        abstained = False
        ranked = search_results
        if not getattr(args, "no_rerank", False):
            reranked = rerank_memory_hits(sample.question, search_results, args.abstain_threshold)
            abstained = reranked is None
            ranked = reranked or search_results
        latency_ms = (time.perf_counter() - started) * 1000

        # Map retrieved memories to session_ids via evidence events
        # Each memory version has citations -> event_ids -> stream_id
        session_order: list[str] = []
        seen: set[str] = set()
        retrieved_memories_detailed: list[dict[str, Any]] = []
        # Build event_id -> session_id cache
        event_session_cache: dict[str, str] = {}

        def _session_for_event(event_id: str) -> str | None:
            if event_id in event_session_cache:
                return event_session_cache[event_id]
            row = engine.database.execute(
                "SELECT stream_id, session_id FROM events WHERE id=? AND namespace_id=?", (event_id, ns)
            ).fetchone()
            if row:
                sid = row["stream_id"] or row["session_id"] or ""
                event_session_cache[event_id] = sid
                return sid
            return None

        for hit in ranked:
            # Gather sessions for this memory from citations
            sessions_for_hit: list[str] = []
            for c in hit.citations:
                sid = _session_for_event(str(c.event_id))
                if sid:
                    sessions_for_hit.append(sid)
            # Fallback: try repository history? Use citations already
            primary_session = sessions_for_hit[0] if sessions_for_hit else ""
            retrieved_memories_detailed.append(
                {
                    "memory_id": str(hit.memory_id),
                    "memory_version_id": str(hit.memory_version_id),
                    "statement": hit.statement,
                    "kind": hit.kind,
                    "score": hit.score,
                    "lexical_score": hit.lexical_score,
                    "vector_score": hit.vector_score,
                    "status": hit.status,
                    "citations": [{"event_id": str(c.event_id), "excerpt": c.excerpt} for c in hit.citations],
                    "evidence_sessions": sessions_for_hit,
                    "primary_session": primary_session,
                }
            )
            # Session aggregation: ordered unique sessions
            for sid in sessions_for_hit:
                if sid not in seen:
                    seen.add(sid)
                    session_order.append(sid)
            # Also include primary session if not already
            if primary_session and primary_session not in seen:
                seen.add(primary_session)
                session_order.append(primary_session)

        # If no citations sessions, we can't map; treat as missed
        packed_text = ""
        if not abstained:
            # Use context building for token budget reporting
            ctx = engine.context(ns, sample.question, token_budget=args.token_budget, limit=args.pack_atoms)
            packed_text = ctx.text

        # If single_db, session_order already isolated via namespace
        oracle = {str(v).strip() for v in sample.answer_session_ids}
        best_rank: int | None = None
        for position, session_id in enumerate(session_order[: args.recall_k], 1):
            if session_id in oracle and (best_rank is None or position < best_rank):
                best_rank = position
        dcg = sum(1.0 / _log2(position + 1) for position, session_id in enumerate(session_order[: args.recall_k], 1) if session_id in oracle)
        idcg = sum(1.0 / _log2(position + 1) for position in range(1, min(len(oracle), args.recall_k) + 1))
        ndcg = dcg / idcg if idcg else 0.0

        # Heuristic failure decomposition
        # Check if any retrieved memory originates from oracle session
        oracle_extracted = False
        # Check via all memories (not just retrieved)
        all_mems = engine.memories(ns, limit=1000)
        oracle_memories_exist = False
        for m in all_mems:
            for c in m.citations:
                sid = _session_for_event(str(c.event_id))
                if sid and sid in oracle:
                    oracle_memories_exist = True
                    break
        # Whether retrieval missed despite existence
        retrieval_missed = oracle_memories_exist and best_rank is None

        # Collect all memories statements for failure analysis
        all_memories_summary = [
            {"memory_id": str(m.memory_id), "statement": m.statement, "kind": m.kind, "status": m.status, "citations": [{"event_id": str(c.event_id)} for c in m.citations]}
            for m in all_mems
        ]

        return {
            "session_order": session_order,
            "best_rank": best_rank,
            "ndcg": ndcg,
            "abstained": abstained,
            "packed": packed_text,
            "packed_words": len(packed_text.split()),
            "latency_ms": round(latency_ms, 2),
            "candidate_count": len(search_results),
            "retrieved_memories": retrieved_memories_detailed,
            "all_memories": all_memories_summary,
            "oracle_memories_exist": oracle_memories_exist,
            "retrieval_missed": retrieval_missed,
            "oracle_extracted": oracle_extracted or oracle_memories_exist,
        }
    finally:
        engine.close()


def retrieve_session_ranking(database_path: Path, sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    db = Database(database_path)
    started = time.perf_counter()
    try:
        limit = max(args.recall_k * 10, 50)
        ns = sample.question_id if getattr(args, "single_db", False) else None
        if args.no_dense:
            hits = search_atoms(db, sample.question, limit, vector_search=lambda *_: [], namespace_id=ns)
        else:
            hits = search_atoms(
                db, sample.question, limit, vector_search=lambda query, lim: shared_dense(db, query, lim, namespace_id=ns), namespace_id=ns
            )
        ranked = hits
        abstained = False
        if not args.no_rerank:
            reranked = rerank_hits(sample.question, hits, args.abstain_threshold)
            abstained = reranked is None
            ranked = reranked or hits
        latency_ms = (time.perf_counter() - started) * 1000
        session_order: list[str] = []
        seen: set[str] = set()
        for hit in ranked:
            if hit.session_id not in seen:
                seen.add(hit.session_id)
                session_order.append(str(hit.session_id))
        packed = "" if abstained else pack_context(ranked[: args.pack_atoms], token_budget=args.token_budget)
        oracle = {str(value).strip() for value in sample.answer_session_ids}
        best_rank: int | None = None
        for position, session_id in enumerate(session_order[: args.recall_k], 1):
            if session_id in oracle and (best_rank is None or position < best_rank):
                best_rank = position
        dcg = sum(1.0 / _log2(position + 1) for position, session_id in enumerate(session_order[: args.recall_k], 1) if session_id in oracle)
        idcg = sum(1.0 / _log2(position + 1) for position in range(1, min(len(oracle), args.recall_k) + 1))
        return {
            "session_order": session_order,
            "best_rank": best_rank,
            "ndcg": dcg / idcg if idcg else 0.0,
            "abstained": abstained,
            "packed": packed,
            "packed_words": len(packed.split()),
            "latency_ms": round(latency_ms, 2),
            "candidate_count": len(hits),
        }
    finally:
        db.close()


def _log2(value: int) -> float:
    return math.log2(max(2, value))


class BudgetExceeded(RuntimeError):
    pass


class OpenRouterBudget:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def charge(self, cost: float) -> None:
        with self._lock:
            self.spent_usd += cost
            if self.spent_usd > self.cap_usd:
                raise BudgetExceeded(f"budget ${self.cap_usd:.2f} exceeded (${self.spent_usd:.4f} spent)")


def openrouter_chat(model: str, messages: list[dict[str, str]], budget: OpenRouterBudget | None, *, max_retries: int = 4) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for judged mode")
    payload = json.dumps({"model": model, "messages": messages}).encode()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            request = Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode())
            usage = body.get("usage", {})
            budget.charge(float(usage.get("cost", 0.0)))
            return body["choices"][0]["message"]["content"]
        except BudgetExceeded:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt + attempt, 15))
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_error}")


ANSWER_SYSTEM = (
    "You answer questions strictly using the provided conversation memory. "
    "Quote exact values when present. If the memory does not contain the answer, reply exactly: insufficient information."
)

JUDGE_SYSTEM = (
    "You are an evaluator for a conversational memory benchmark. Given a question, a reference answer, "
    "and a system response, decide whether the response conveys the reference answer. "
    'Reply with JSON only: {"correct": true} or {"correct": false}. Unanswerable questions are correct '
    "only when the response says it lacks the information."
)


def judge_question(question_model: str, judge_model: str, sample: Sample, context: str, budget: OpenRouterBudget) -> dict[str, Any]:
    if sample.unanswerable or context.strip() == "":
        hypothesis = "(no context provided)"
    else:
        hypothesis = openrouter_chat(
            question_model,
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {sample.question}"},
            ],
            budget,
        )
    verdict_raw = openrouter_chat(
        judge_model,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {sample.question}\nReference answer: {sample.answer}\n"
                    f"Question is answerable: {not sample.unanswerable}\nSystem response: {hypothesis}"
                ),
            },
        ],
        budget,
    )
    try:
        correct = bool(json.loads(verdict_raw.strip().strip("`").removeprefix("json"))["correct"])
    except Exception:
        correct = "true" in verdict_raw.lower()
    return {"hypothesis": hypothesis, "judge_verdict": verdict_raw, "correct": correct}


def evaluate_sample(args: argparse.Namespace, sample: Sample, budget: OpenRouterBudget | None) -> dict[str, Any]:
    database_path = ingest_sample(
        Path(args.work_dir), sample, skip_embeddings=args.no_dense, single_db=getattr(args, "single_db", False)
    )
    outcome = retrieve_session_ranking(database_path, sample, args)
    trace: dict[str, Any] = {
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "answer_session_ids": sorted(sample.answer_session_ids),
        "best_rank": outcome["best_rank"],
        "ndcg_at_k": round(outcome["ndcg"], 4),
        "abstained": outcome["abstained"],
        "recall": {
            str(k): int(outcome["best_rank"] is not None and outcome["best_rank"] <= k)
            for k in (5, 10, args.recall_k)
        },
        "packed_words": outcome["packed_words"],
        "raw_words": sample.raw_words,
        "retrieval_latency_ms": outcome["latency_ms"],
        "candidate_count": outcome["candidate_count"],
    }
    if args.mode == "judged" and budget is not None:
        judged = judge_question(args.answer_model, args.judge_model, sample, outcome["packed"], budget)
        trace.update(judged)
    return trace


def evaluate_sample_e2e(args: argparse.Namespace, sample: Sample, budget: OpenRouterBudget | None) -> dict[str, Any]:
    """Evaluate one sample through production pipeline.

    Steps (with leakage boundary):
      1. ingest_and_process_e2e  uses ONLY haystack_sessions (no question/answer)
      2. retrieve_e2e_session_ranking  uses question ONLY for retrieval
    """
    work_dir = Path(args.work_dir)
    ingest_start = time.perf_counter()
    db_path, e2e_diag = ingest_and_process_e2e(work_dir, sample, args)
    total_e2e_ms = (time.perf_counter() - ingest_start) * 1000
    retrieval_outcome = retrieve_e2e_session_ranking(db_path, sample, args)
    # Failure decomposition heuristics
    best_rank = retrieval_outcome["best_rank"]
    oracle_exist = retrieval_outcome.get("oracle_memories_exist", False)
    retrieval_missed = retrieval_outcome.get("retrieval_missed", False)
    # Classify failure reason
    if best_rank is not None:
        failure_reason = "none"
    elif not oracle_exist:
        # No memory from oracle session was ever created
        if e2e_diag.get("candidates_accepted", 0) == 0 and e2e_diag.get("candidates_rejected", 0) > 0:
            failure_reason = "candidate_rejected"
        elif e2e_diag.get("candidates_extracted", 0) == 0:
            failure_reason = "never_extracted"
        else:
            failure_reason = "never_extracted_or_rejected"
    elif retrieval_missed:
        failure_reason = "memory_existed_retrieval_missed"
    else:
        # Check token budget / ranking miss
        if retrieval_outcome.get("abstained"):
            failure_reason = "abstained"
        else:
            failure_reason = "context_budget_or_ranking_miss"

    # Build oracle session texts for failure analysis (without leaking into extraction)
    oracle_texts = []
    for sid, _, turns in sample.sessions:
        if sid in sample.answer_session_ids:
            oracle_texts.append({"session_id": sid, "turns": [{"role": t["role"], "content": t["content"][:500]} for t in turns]})

    trace: dict[str, Any] = {
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "question": sample.question,
        "answer": sample.answer,
        "answer_session_ids": sorted(sample.answer_session_ids),
        "oracle_session_texts": oracle_texts,
        "best_rank": best_rank,
        "ndcg_at_k": round(retrieval_outcome["ndcg"], 4),
        "abstained": retrieval_outcome["abstained"],
        "recall": {
            str(k): int(best_rank is not None and best_rank <= k)
            for k in (5, 10, args.recall_k)
        },
        "packed_words": retrieval_outcome["packed_words"],
        "raw_words": sample.raw_words,
        "retrieval_latency_ms": retrieval_outcome["latency_ms"],
        "candidate_count": retrieval_outcome["candidate_count"],
        # Memory-formation diagnostics
        "e2e_diagnostics": e2e_diag,
        "events_ingested": e2e_diag.get("events_ingested", 0),
        "processing_jobs_completed": e2e_diag.get("processing_jobs_completed", 0),
        "processing_jobs_failed": e2e_diag.get("processing_jobs_failed", 0),
        "processing_jobs_dead": e2e_diag.get("processing_jobs_dead", 0),
        "candidates_extracted": e2e_diag.get("candidates_extracted", 0),
        "candidates_accepted": e2e_diag.get("candidates_accepted", 0),
        "candidates_rejected": e2e_diag.get("candidates_rejected", 0),
        "memories_created": e2e_diag.get("memories_created", 0),
        "rejection_reasons": e2e_diag.get("rejection_reasons", {}),
        "total_e2e_latency_ms": round(total_e2e_ms, 2),
        "extraction_latency_ms": e2e_diag.get("extraction_latency_ms", 0),
        # Retrieval details for failure analysis
        "retrieved_memories": retrieval_outcome.get("retrieved_memories", [])[:15],
        "all_memories": retrieval_outcome.get("all_memories", [])[:30],
        "session_order": retrieval_outcome.get("session_order", [])[:15],
        "oracle_memories_exist": oracle_exist,
        "retrieval_missed": retrieval_missed,
        "failure_reason": failure_reason,
        "failure_analysis": {
            "question": sample.question,
            "category": sample.question_type,
            "relevant_source_evidence": oracle_texts,
            "memories_actually_produced": retrieval_outcome.get("all_memories", [])[:30],
            "retrieved_memories": retrieval_outcome.get("retrieved_memories", [])[:15],
            "session_order": retrieval_outcome.get("session_order", []),
            "best_rank": best_rank,
            "oracle_memories_exist": oracle_exist,
            "retrieval_missed": retrieval_missed,
            "failure_reason": failure_reason,
            "diagnostics": e2e_diag,
        },
    }
    if args.mode == "judged" and budget is not None:
        judged = judge_question(args.answer_model, args.judge_model, sample, retrieval_outcome["packed"], budget)
        trace.update(judged)
    return trace


def summarize(traces: list[dict[str, Any]], recall_k: int, judged: bool) -> list[dict[str, Any]]:
    ks = ["5", "10", str(recall_k)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["question_type"]].append(trace)
    rows: list[dict[str, Any]] = []

    def row_for(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        row: dict[str, Any] = {"Category": name, "n": count}
        for k in ks:
            row[f"Recall@{k} (%)"] = round(100 * sum(item["recall"][k] for item in items) / count, 1) if count else 0.0
        row[f"MRR@{recall_k}"] = round(sum((1 / item["best_rank"]) if item["best_rank"] else 0.0 for item in items) / count, 3) if count else 0.0
        row[f"NDCG@{recall_k}"] = round(sum(item["ndcg_at_k"] for item in items) / count, 3) if count else 0.0
        row["Avg Context Tokens"] = round(sum(item["packed_words"] for item in items) / count, 1) if count else 0.0
        row["Avg Latency (ms)"] = round(sum(item["retrieval_latency_ms"] for item in items) / count, 1) if count else 0.0
        # End-to-end diagnostics averages where present
        if any("e2e_diagnostics" in it for it in items):
            row["Avg Memories"] = round(sum(it.get("memories_created", 0) for it in items) / count, 1) if count else 0.0
            row["Avg Candidates"] = round(sum(it.get("candidates_extracted", 0) for it in items) / count, 1) if count else 0.0
        if judged:
            row["Judged Acc (%)"] = round(100 * sum(int(bool(item.get("correct"))) for item in items) / count, 1) if count else 0.0
        return row

    for category in CATEGORY_ORDER:
        if category in grouped:
            rows.append(row_for(category, grouped[category]))
    rows.append(row_for("Overall", traces))
    return rows


def failure_decomposition(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(t.get("failure_reason", "unknown") for t in traces if t.get("best_rank") is None)
    return {"counts": dict(counter), "total_missed": sum(counter.values()), "total": len(traces)}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return None


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    data_path = Path(args.data_path)
    dataset_bytes = data_path.read_bytes()
    dataset_sha256 = hashlib_sha256(dataset_bytes)
    samples = normalize_samples(json.loads(dataset_bytes.decode("utf-8")))
    if args.task:
        samples = [item for item in samples if item.question_type == args.task]
    resume_ids: set[str] = set()
    previous_traces: list[dict[str, Any]] = []
    if args.resume_from:
        previous = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        previous_traces = previous.get("traces", [])
        resume_ids = {trace["question_id"] for trace in previous_traces}
        print(f"Resuming: {len(resume_ids)} already complete", flush=True)
    pending = [item for item in samples if item.question_id not in resume_ids]
    if args.limit:
        pending = pending[: args.limit]
    if args.smoke:
        if not args.confirm_benchmark:
            raise SystemExit("smoke benchmark loops require --confirm-benchmark")
        pending = pending[: args.smoke_samples]

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    # Normalize mode aliases
    raw_mode = args.mode
    if raw_mode in ("retrieval", "retrieval-only"):
        canonical_mode = "retrieval-only"
    elif raw_mode in ("end-to-end", "end_to_end", "e2e", "end-to-end-retrieval"):
        canonical_mode = "end-to-end"
    elif raw_mode == "judged":
        canonical_mode = "judged"
    else:
        canonical_mode = raw_mode

    is_e2e = canonical_mode == "end-to-end"
    if is_e2e:
        if getattr(args, "extraction", "openrouter") != "openrouter":
            raise SystemExit("end-to-end benchmark requires --extraction openrouter")
        if getattr(args, "embedding_provider", None) not in (None, "openrouter"):
            raise SystemExit("end-to-end benchmark requires --embedding-provider openrouter")
        if getattr(args, "embedding_provider", None) is None:
            args.embedding_provider = "openrouter"
    budget = OpenRouterBudget(args.budget_usd) if canonical_mode == "judged" else None

    manifest = _run_manifest(args, data_path, dataset_sha256, canonical_mode)
    manifest_path = _run_manifest_path(work_dir)
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest != manifest:
            raise SystemExit(
                f"work dir {work_dir} already has a different LongMemEval run manifest; "
                "use a fresh work dir or pass --resume-from for the same run"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    traces: list[dict[str, Any]] = list(previous_traces)
    failures = 0
    started = time.perf_counter()

    def worker_retrieval(sample: Sample) -> dict[str, Any]:
        return evaluate_sample(args, sample, budget)

    def worker_e2e(sample: Sample) -> dict[str, Any]:
        return evaluate_sample_e2e(args, sample, budget)

    worker = worker_e2e if is_e2e else worker_retrieval

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, sample): sample for sample in pending}
        for number, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            try:
                trace = future.result()
                traces.append(trace)
                status = f"rank={trace['best_rank']}"
                if "correct" in trace:
                    status += f"; judged={'correct' if trace['correct'] else 'wrong'}"
                # Add e2e extra info
                if is_e2e:
                    diag = trace.get("e2e_diagnostics", {})
                    status += f"; mems={diag.get('memories_created',0)} acc={diag.get('candidates_accepted',0)} rej={diag.get('candidates_rejected',0)}"
                print(f"[{number}/{len(pending)}] {sample.question_id} ({sample.question_type}): {status}; {trace['retrieval_latency_ms']:.0f}ms", flush=True)
            except BudgetExceeded as exc:
                failures += 1
                print(f"BUDGET STOP: {exc}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as exc:
                failures += 1
                print(f"[{number}/{len(pending)}] {sample.question_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
                import traceback as _tb
                _tb.print_exc()

    rows = summarize(traces, args.recall_k, judged=canonical_mode == "judged")
    table = render_table(rows)
    git_commit = _git_commit()
    # Embedding provider info
    embed_name = shared_product_embedder(args).name if is_e2e else shared_embedder().name
    # Extraction provider info
    extraction_provider = getattr(args, "extraction", "rule") if is_e2e else "verbatim-atoms"
    extraction_model = getattr(args, "extraction_model", None)
    result = {
        "mode": canonical_mode,
        "raw_mode": raw_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "dataset": {"path": str(data_path), "sha256": dataset_sha256, "samples_total": len(samples)},
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "failures": failures,
        "budget_spent_usd": round(budget.spent_usd, 4) if budget else 0.0,
        "summary": rows,
        "traces": traces,
        "failure_decomposition": failure_decomposition(traces) if is_e2e else None,
        "run_metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "benchmark_mode": canonical_mode,
            "dataset_sha256": dataset_sha256,
            "question_count": len(traces),
            "extraction_provider": extraction_provider,
            "extraction_model": str(extraction_model) if extraction_model else None,
            "embedding_provider": embed_name,
            "reranker": "ms-marco-MiniLM-L-12-v2" if not args.no_rerank else None,
            "dense_enabled": not args.no_dense,
            "workers": args.workers,
            "token_budget": args.token_budget,
            "top_k_values": [5, 10, args.recall_k],
            "recall_k": args.recall_k,
            "pack_atoms": args.pack_atoms,
            "abstain_threshold": args.abstain_threshold,
        },
    }
    output_path = Path(args.results_dir) / f"longmemeval_s_{canonical_mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + table)
    if is_e2e and result.get("failure_decomposition"):
        print("\nFailure decomposition:", json.dumps(result["failure_decomposition"], indent=2))
    print(f"\nTraces: {output_path}")
    if budget:
        print(f"Spend: ${budget.spent_usd:.4f}")
    return 0


def hashlib_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="TermyteDB LongMemEval-S benchmark")
    parser.add_argument("--mode", choices=("retrieval", "retrieval-only", "end-to-end", "judged", "end_to_end", "e2e"), default="end-to-end", help="Benchmark pipeline: retrieval-only (verbatim atoms) or end-to-end (production events)")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--work-dir", default=str(ROOT / ".termytedb-work" / "longmemeval"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--limit", type=int, help="limit number of questions (for smoke tests)")
    parser.add_argument("--smoke", action="store_true", help="run the manual 5-sample benchmark smoke subset")
    parser.add_argument("--smoke-samples", type=int, default=5, help="number of questions to use for smoke runs")
    parser.add_argument("--confirm-benchmark", action="store_true", help="required to start a benchmark smoke loop")
    parser.add_argument("--task", choices=CATEGORY_ORDER, help="filter to single question_type")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recall-k", type=int, default=15)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--pack-atoms", type=int, default=40, help="max atoms/memories to pack into context")
    parser.add_argument("--abstain-threshold", type=float, default=0.25)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--single-db", action="store_true", help="store all questions in one SQLite file (namespace-isolated)")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--answer-model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--budget-usd", type=float, default=8.0)
    parser.add_argument("--embedding-provider", choices=("local", "openrouter"), default=None, help="embedding provider for end-to-end runs")
    parser.add_argument("--embedding-model", type=str, default=None, help="model for OpenRouter-compatible embeddings")
    parser.add_argument("--embedding-dimensions", type=int, default=None, help="dimensions for OpenRouter-compatible embeddings")
    # End-to-end extraction config
    parser.add_argument("--extraction", choices=("rule", "openrouter", "fake", "http"), default="openrouter", help="extraction provider for end-to-end mode (OpenRouter is the product default)")
    parser.add_argument("--extraction-model", type=str, default=None, help="model for openrouter/http extraction (or env TERMYTEDB_EXTRACTION_MODEL)")
    parser.add_argument("--processing-batch-size", type=int, default=100, help="processing jobs per batch")
    parser.add_argument("--processing-lease-seconds", type=int, default=180)
    parser.add_argument("--processing-timeout", type=float, default=30.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

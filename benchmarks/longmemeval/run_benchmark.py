"""LongMemEval-S benchmark for TermyteDB.

Canonical harness supporting two distinct benchmark pipelines:

* retrieval-only (``retrieval`` / ``retrieval-only``) — lossless episodic retrieval
  isolation benchmark: verbatim turn-level atoms -> FTS5 + dense hybrid (RRF) ->
  FlashRank rerank -> session aggregation -> retrieval metrics. This is the
  existing upper-bound benchmark; it shortcuts memory formation and measures
  indexing + retrieval only.

* end-to-end (``end-to-end``) — true production memory pipeline:
  LongMemEval conversation/session history -> direct EventInput ingestion ->
  extraction (OpenRouter / fake / HTTP) -> evidence validation
  -> reconciliation/versioning -> embeddings/indexing -> hybrid retrieval
  -> session aggregation -> retrieval metrics.

  No ground-truth question, answer, answer aliases, relevance annotations,
  evidence identifiers, or category hints leak into memory formation.
  The question appears only after ingestion completes.

General methodology notes are in docs/benchmarks.md; additional pipeline
details for end-to-end are in the docstring of ``ingest_e2e``
and ``evaluate_sample_e2e``.

Modes:
  retrieval / retrieval-only  Zero-cost session-level retrieval (atoms)
  end-to-end                 Production pipeline via TermyteDB events/memories
  judged                     retrieval-only + OpenRouter answer generation/judging
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.embedding import FastEmbedProvider  # noqa: E402
from src.retrieval.retrieval import AtomHit, dense_search_atoms, search_atoms  # noqa: E402
from src.storage.db import Database  # noqa: E402

DEFAULT_DATA_PATH = ROOT / "benchmarks" / "longmemeval" / "longmemeval_s_cleaned.json"
DEFAULT_MICRO_PATH = ROOT / "benchmarks" / "longmemeval" / "longmemeval_micro.json"
CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
]


@dataclass(frozen=True)
class L1Atom:
    atom_id: str
    session_id: str
    fact: str
    timestamp: str | None
    source_role: str
    namespace_id: str | None = None


def insert_atoms(db: Database, atoms: list[L1Atom]) -> None:
    now = datetime.now(UTC).isoformat()
    with db.connection:
        db.connection.executemany(
            """INSERT OR IGNORE INTO atoms
               (atom_id, session_id, fact, timestamp, source_role, created_at, namespace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(a.atom_id, a.session_id, a.fact.strip(), a.timestamp, a.source_role, now, a.namespace_id) for a in atoms],
        )


def index_atom_embeddings(db: Database, provider: Any, *, batch_size: int = 64) -> None:
    rows = db.execute(
        """SELECT a.atom_id, a.fact FROM atoms a
           LEFT JOIN atom_embeddings e ON e.atom_id=a.atom_id AND e.provider=?
           WHERE e.atom_id IS NULL ORDER BY a.rowid""",
        (provider.name,),
    ).fetchall()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = provider.embed_many([str(row["fact"]) for row in batch])
        with db.connection:
            db.connection.executemany(
                "INSERT OR REPLACE INTO atom_embeddings(atom_id, provider, dimensions, vector) VALUES (?, ?, ?, ?)",
                [(row["atom_id"], provider.name, len(vector), array.array("f", vector).tobytes()) for row, vector in zip(batch, vectors, strict=True)],
            )


_ranker = None
_ranker_lock = threading.Lock()
_embedder: GuardedEmbedder | None = None
_product_embedder: GuardedEmbedder | None = None
_product_embedder_key: tuple[str, str | None, int | None] | None = None
_product_embedder_lock = threading.Lock()
_RUN_MANIFEST_NAME = "longmemeval_run_manifest.json"
_CHECKPOINT_NAME = "longmemeval_checkpoint.json"
_BENCHMARK_LOGGER = logging.getLogger("longmemeval.benchmark")


def _configure_benchmark_logging(log_path: Path, *, append: bool) -> None:
    """Keep the terminal concise while retaining full engine diagnostics."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    termyte_logger = logging.getLogger("termytedb")
    termyte_logger.setLevel(logging.INFO)
    # The library logger normally prints every accepted event to stderr. Keep
    # that detail in the run log instead; progress belongs to the CLI.
    for handler in termyte_logger.handlers:
        handler.setLevel(logging.CRITICAL + 1)
    for handler in list(termyte_logger.handlers):
        if getattr(handler, "_longmemeval_log", False):
            termyte_logger.removeHandler(handler)
            handler.close()
    mode = "a" if append else "w"
    engine_file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    engine_file_handler._longmemeval_log = True  # type: ignore[attr-defined]
    engine_file_handler.setLevel(logging.INFO)
    engine_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    termyte_logger.addHandler(engine_file_handler)

    _BENCHMARK_LOGGER.setLevel(logging.INFO)
    _BENCHMARK_LOGGER.propagate = False
    for handler in list(_BENCHMARK_LOGGER.handlers):
        _BENCHMARK_LOGGER.removeHandler(handler)
        handler.close()
    runner_file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    runner_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _BENCHMARK_LOGGER.addHandler(runner_file_handler)


class RequestPacer:
    """Process-wide minimum spacing for remote provider requests."""

    def __init__(self, min_interval_seconds: float = 2.0) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            if delay:
                time.sleep(delay)
            self._next_request_at = time.monotonic() + self.min_interval_seconds
            return delay

    def defer(self, delay_seconds: float) -> None:
        with self._lock:
            self._next_request_at = max(self._next_request_at, time.monotonic() + max(0.0, delay_seconds))


_openrouter_pacer = RequestPacer()


class RateLimitedExtractionProvider:
    def __init__(self, inner: Any, *, max_retries: int, rate_limit_cooldown: float = 60.0) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.max_retries = max(1, max_retries)
        self.rate_limit_cooldown = max(0.0, rate_limit_cooldown)

    def extract(self, request: Any, timeout_seconds: float = 30.0, cancellation: Any = None) -> Any:
        from src.memory.provider import ProviderError  # noqa: E402

        last_error: ProviderError | None = None
        for attempt in range(self.max_retries):
            waited = _openrouter_pacer.wait()
            if waited:
                _BENCHMARK_LOGGER.info("OpenRouter pacing: waited %.1fs", waited)
            try:
                return self.inner.extract(request, timeout_seconds=timeout_seconds, cancellation=None)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.max_retries - 1:
                    raise
                exponential_delay = min(30.0, (2**attempt) + random.uniform(0.0, 1.0))
                delay = max(self.rate_limit_cooldown, exponential_delay) if "HTTP 429" in str(exc) else exponential_delay
                _BENCHMARK_LOGGER.warning("OpenRouter retry %s/%s after %.1fs: %s", attempt + 1, self.max_retries - 1, delay, exc)
                _openrouter_pacer.defer(delay)
        raise RuntimeError("OpenRouter extraction retry loop ended unexpectedly") from last_error


class GuardedEmbedder:
    """Serializes ONNX inference so concurrent workers cannot grow competing arenas."""

    def __init__(self, inner: Any, *, pace_remote: bool = False) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._pace_remote = pace_remote

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, value: str) -> list[float]:
        with self._lock:
            if self._pace_remote:
                _openrouter_pacer.wait()
            return self._inner.embed(value)

    def embed_many(self, values: list[str]) -> list[list[float]]:
        with self._lock:
            if self._pace_remote:
                _openrouter_pacer.wait()
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
                    from src.retrieval.embedding import OpenAICompatibleEmbeddingProvider  # noqa: E402

                    _product_embedder = GuardedEmbedder(OpenAICompatibleEmbeddingProvider(model_name, dimensions=dimensions), pace_remote=True)
                else:
                    _product_embedder = shared_embedder()
                _product_embedder_key = key
    return _product_embedder


_rerank_mutex = threading.Lock()


def _rerank(query: str, hits: list[Any], threshold: float, *, item_id: Any, item_text: Any, max_candidates: int = 30, max_chars: int = 600) -> list[Any] | None:
    from flashrank import RerankRequest  # type: ignore[import-untyped]

    ranker = shared_ranker()
    candidates = hits[:max_candidates]
    tail = hits[max_candidates:]
    passages = [{"id": item_id(hit), "text": item_text(hit)[:max_chars]} for hit in candidates]
    with _rerank_mutex:
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    scored = [(entry["id"], float(entry.get("score", 0.0))) for entry in results]
    by_id = {item_id(hit): hit for hit in candidates}
    ordered = [by_id[result_id] for result_id, _ in scored if result_id in by_id]
    if not ordered or scored[0][1] < threshold:
        return None
    return ordered + tail


def rerank_hits(query: str, hits: list[AtomHit], threshold: float, *, max_candidates: int = 30, max_chars: int = 600) -> list[AtomHit] | None:
    """Shared-instance FlashRank rerank with hard abstention."""
    return _rerank(query, hits, threshold, item_id=lambda hit: hit.atom_id, item_text=lambda hit: hit.fact, max_candidates=max_candidates, max_chars=max_chars)


def rerank_memory_hits(query: str, hits: list[Any], threshold: float, *, max_candidates: int = 30, max_chars: int = 600) -> list[Any] | None:
    """FlashRank rerank for TermyteDB SearchResult objects."""
    return _rerank(
        query,
        hits,
        threshold,
        item_id=lambda hit: str(hit.memory_version_id),
        item_text=lambda hit: hit.statement,
        max_candidates=max_candidates,
        max_chars=max_chars,
    )


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    question_date: str
    question_type: str
    answer: str
    answer_session_ids: frozenset[str]
    unanswerable: bool
    sessions: tuple[tuple[str, str, tuple[dict[str, str], ...]], ...]
    raw_words: int


def normalize_samples(raw: Any) -> list[Sample]:
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("samples", []))
    if not isinstance(items, list):
        raise ValueError("dataset must contain a list of samples")
    samples: list[Sample] = []
    for item in items:
        required = ("question_id", "question", "question_type", "answer", "answer_session_ids", "haystack_session_ids", "haystack_dates", "haystack_sessions")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"sample missing required fields: {', '.join(missing)}")
        sessions: list[tuple[str, str, tuple[dict[str, str], ...]]] = []
        raw_words = 0
        ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])
        histories = item.get("haystack_sessions", [])
        if not (len(ids) == len(dates) == len(histories)):
            raise ValueError(f"{item['question_id']}: haystack arrays must have matching lengths")
        for index, messages in enumerate(histories):
            if not isinstance(ids[index], str) or not ids[index].strip():
                raise ValueError(f"{item['question_id']}: session IDs must be non-empty strings")
            session_id = ids[index]
            date = str(dates[index])
            if not isinstance(messages, list):
                raise ValueError(f"{item['question_id']}: session turns must be lists")
            turns = tuple(
                {"role": message["role"], "content": message["content"]} for message in messages
            )
            if any(turn["role"] not in ("user", "assistant") or not isinstance(turn["content"], str) for turn in turns):
                raise ValueError(f"{item['question_id']}: turns require user/assistant role and string content")
            raw_words += sum(len(turn["content"].split()) for turn in turns)
            sessions.append((session_id, date, turns))
        dated = [(_parse_haystack_date(session[1]), index, session) for index, session in enumerate(sessions)]
        if any(parsed is not None for parsed, _, _ in dated):
            sessions = [session for _, _, session in sorted(dated, key=lambda value: (value[0] is None, value[0] or datetime.min.replace(tzinfo=UTC), value[1]))]
        # LongMemEval-S question timestamps are metadata, not an ingestion
        # cutoff. The supplied dataset can include haystack sessions later on
        # the same day, so rejecting those samples makes the standard dataset
        # impossible to run. We still order sessions by their actual dates.
        samples.append(
            Sample(
                question_id=str(item.get("question_id", len(samples))),
                question=str(item.get("question", "")),
                question_date=str(item.get("question_date", "")),
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


def ingest_sample(work_dir: Path, sample: Sample, *, skip_embeddings: bool = False, single_db: bool = False) -> Path:
    if single_db:
        database_path = work_dir / "single.sqlite"
        # Serialize writes to the single file to avoid WAL contention
        with _single_db_lock:
            db = Database(database_path)
            try:
                _ensure_namespace(db, sample.question_id)
                existing = db.execute("SELECT COUNT(*) FROM atoms WHERE namespace_id = ?", (sample.question_id,)).fetchone()[0]
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
        from src.memory.provider import OpenRouterExtractionProvider  # noqa: E402

        model = getattr(args, "extraction_model", None) or os.environ.get("TERMYTEDB_EXTRACTION_MODEL")
        if not model:
            raise ValueError("TERMYTEDB_EXTRACTION_MODEL or --extraction-model is required")
        provider = OpenRouterExtractionProvider(model=model)
        return RateLimitedExtractionProvider(
            provider,
            max_retries=int(getattr(args, "openrouter_max_retries", 5)),
            rate_limit_cooldown=float(getattr(args, "openrouter_rate_limit_cooldown", 60.0)),
        )
    if name == "rule":
        from src.memory.provider import FakeExtractionProvider  # noqa: E402

        return FakeExtractionProvider()
    if name == "fake":
        from src.memory.provider import FakeExtractionProvider  # noqa: E402

        return FakeExtractionProvider()
    if name == "http":
        from src.memory.provider import HttpExtractionProvider  # noqa: E402

        return HttpExtractionProvider()
    raise ValueError(f"unknown extraction provider: {name}")


def _e2e_database_path(work_dir: Path, sample: Sample, single_db: bool) -> Path:
    if single_db:
        return work_dir / "e2e_single.sqlite"
    return work_dir / f"e2e_{sample.question_id}.sqlite"


def _run_manifest_path(work_dir: Path) -> Path:
    return work_dir / _RUN_MANIFEST_NAME


def _checkpoint_path(work_dir: Path) -> Path:
    return work_dir / _CHECKPOINT_NAME


def _write_checkpoint(
    work_dir: Path,
    *,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    dataset_sha256: str,
    traces: list[dict[str, Any]],
) -> None:
    """Persist completed traces so Ctrl+C never discards an expensive run."""
    checkpoint = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_sha256,
        "manifest": manifest,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "traces": traces,
    }
    path = _checkpoint_path(work_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_work_dir(args: argparse.Namespace, previous: dict[str, Any] | None, *, is_e2e: bool) -> Path:
    base = Path(args.work_dir)
    if not is_e2e:
        return base
    resume_work_dir = getattr(args, "resume_work_dir", None)
    if resume_work_dir:
        return Path(resume_work_dir)
    if previous is not None:
        prior_work_dir = previous.get("config", {}).get("work_dir")
        if not prior_work_dir:
            raise ValueError("resume result does not contain its work directory")
        return Path(prior_work_dir)
    run_id = f"{datetime.now(UTC).strftime('run-%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    return base / run_id


def _get_retrieval_limit(args: argparse.Namespace) -> int:
    value = getattr(args, "retrieval_limit", None)
    if value is None:
        value = getattr(args, "pack_atoms", 40)
    return int(value)


def _run_manifest(args: argparse.Namespace, data_path: Path, dataset_sha256: str, canonical_mode: str) -> dict[str, Any]:
    retrieval_limit = _get_retrieval_limit(args)
    return {
        "mode": canonical_mode,
        "baseline": getattr(args, "baseline", "system"),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "extraction": getattr(args, "extraction", None),
        "extraction_model": getattr(args, "extraction_model", None),
        "embedding_provider": getattr(args, "embedding_provider", None),
        "embedding_model": getattr(args, "embedding_model", None),
        "embedding_dimensions": getattr(args, "embedding_dimensions", None),
        "extraction_batch_sessions": int(getattr(args, "extraction_batch_sessions", 4)),
        "single_db": bool(getattr(args, "single_db", False)),
        "no_dense": bool(getattr(args, "no_dense", False)),
        "no_rerank": bool(getattr(args, "no_rerank", False)),
        "recall_k": int(getattr(args, "recall_k", 15)),
        "retrieval_limit": retrieval_limit,
        "abstain_threshold": float(getattr(args, "abstain_threshold", 0.25)),
    }


def _normalize_manifest_for_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy benchmark manifests for resume compatibility.

    Pre-refactor manifests used `pack_atoms` and `token_budget`; post-refactor
    uses `retrieval_limit` and drops `token_budget` (search no longer has token
    budgets). For resume we treat `pack_atoms == retrieval_limit` and ignore
    `token_budget` so existing work dirs remain resumable.
    """
    normalized = dict(manifest)
    # Migrate pack_atoms -> retrieval_limit. If both exist, the canonical key wins.
    if "retrieval_limit" not in normalized:
        if "pack_atoms" in normalized:
            try:
                normalized["retrieval_limit"] = int(normalized["pack_atoms"])
            except Exception:
                normalized["retrieval_limit"] = 40
        else:
            normalized["retrieval_limit"] = 40
    else:
        try:
            normalized["retrieval_limit"] = int(normalized["retrieval_limit"])
        except (TypeError, ValueError):
            normalized["retrieval_limit"] = 40
    # token_budget is obsolete — do not affect run identity
    normalized.pop("token_budget", None)
    # pack_atoms is now an alias; remove legacy key for canonical comparison
    normalized.pop("pack_atoms", None)
    return normalized


def ingest_e2e(work_dir: Path, sample: Sample, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Ingest LongMemEval history through the direct production pipeline.

    Returns (database_path, diagnostics).
    """
    from src import EventInput, TermyteDB  # noqa: E402

    single_db = bool(getattr(args, "single_db", False))
    database_path = _e2e_database_path(work_dir, sample, single_db)
    namespace_id = sample.question_id
    provider = build_provider(args)
    # Build embedding provider: shared guarded provider for reuse
    embedding_provider = shared_product_embedder(args)

    # Use file lock for single-db concurrent access
    lock = _single_db_lock if single_db else threading.Lock()
    # Serialize direct ingestion for a shared SQLite file.
    with lock:
        engine = TermyteDB(database_path, extraction_provider=provider, embedding_provider=embedding_provider)  # type: ignore[arg-type]
        try:
            events_raw = build_event_inputs(sample)
            # Filter to EventInput for validation
            events_input = [EventInput.model_validate(e) for e in events_raw]

            ingest_started = time.perf_counter()
            events_ingested = 0
            events_duplicate = 0
            total_accepted = total_rejected = 0
            receipts = []
            session_batches: dict[str, list[EventInput]] = {}
            for ev in events_input:
                scope = str(ev.session_id or ev.stream_id or ev.idempotency_key)
                session_batches.setdefault(scope, []).append(ev)
            # One provider call handles a small group of complete sessions.
            # This preserves session boundaries while avoiding one remote call
            # per session (the dominant source of benchmark runtime).
            sessions_per_extraction = max(1, int(getattr(args, "extraction_batch_sessions", 4)))
            sessions = list(session_batches.values())
            batches: list[list[EventInput]] = []
            for index in range(0, len(sessions), sessions_per_extraction):
                batches.append([event for session in sessions[index : index + sessions_per_extraction] for event in session])
            for batch in batches:
                result = engine.ingest_batch(batch)
                receipts.extend(result.receipts)
                total_accepted += result.accepted
                total_rejected += result.rejected
                events_duplicate += sum(receipt.duplicate for receipt in result.receipts)
                events_ingested += sum(not receipt.duplicate for receipt in result.receipts)
            # On resume, previously accepted events are duplicates. Process any
            # durable failed jobs left by the interrupted attempt instead of
            # silently treating the sample as complete without extraction.
            retry_result = engine.process(namespace_id, limit=1000)
            total_accepted += retry_result.accepted
            total_rejected += retry_result.rejected
            ingest_latency_ms = (time.perf_counter() - ingest_started) * 1000
            # Collect completed direct-ingestion diagnostics.
            metrics_final = engine.metrics(namespace_id)
            runs = engine.extraction_runs(namespace_id, limit=1000)
            decisions = engine.extraction_decisions(namespace_id, limit=1000)
            memories = engine.memories(namespace_id, limit=1000)
            # Rejection reasons
            rejection_counter = Counter(d.get("rejection_reason") for d in decisions if d.get("validation_status") == "rejected" and d.get("rejection_reason"))
            # If dense disabled, optionally clear embeddings so retrieval becomes lexical-only
            if getattr(args, "no_dense", False):
                try:
                    engine.database.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", (namespace_id,))
                    engine.database.connection.commit()
                except Exception:
                    pass

            diagnostics: dict[str, Any] = {
                "events_ingested": events_ingested,
                "events_duplicate": events_duplicate,
                "events_total": len(events_input),
                "ingest_latency_ms": round(ingest_latency_ms, 2),
                "candidates_extracted": total_accepted + total_rejected,
                "candidates_accepted": total_accepted,
                "candidates_rejected": total_rejected,
                "memories_created": len(memories),
                "memory_versions_created": int(metrics_final.get("memory_versions", 0)),
                "average_memories_per_sample": len(memories),
                "extraction_latency_ms": round(ingest_latency_ms, 2),
                "metrics": metrics_final,
                "rejection_reasons": dict(rejection_counter),
                "runs_count": len(runs),
                "decisions_count": len(decisions),
                "extraction_batches": len(batches),
                "sessions_per_extraction": sessions_per_extraction,
            }
        finally:
            engine.close()
    return database_path, diagnostics


def retrieve_e2e_session_ranking(database_path: Path, sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    """Retrieve after direct ingestion using production memories."""
    from src import TermyteDB  # noqa: E402

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
            ranked = reranked if reranked is not None else []
        # Raw-session fallback: an empty LLM extraction must never make a
        # haystack session invisible to the end-to-end benchmark.
        raw_sessions = engine.search_sessions(ns, sample.question, limit=max(args.recall_k * 2, 30))
        if raw_sessions:
            abstained = False
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
            row = engine.database.execute("SELECT stream_id, session_id FROM events WHERE id=? AND namespace_id=?", (event_id, ns)).fetchone()
            if row:
                sid = row["stream_id"] or row["session_id"] or ""
                event_session_cache[event_id] = sid
                return sid
            return None

        # Put raw sessions first. They are direct source material and make the
        # benchmark answer context useful even when no memory was extracted.
        for raw_session in raw_sessions:
            if raw_session.session_id not in seen:
                seen.add(raw_session.session_id)
                session_order.append(raw_session.session_id)
            retrieved_memories_detailed.append(
                {
                    "memory_id": None,
                    "statement": "Raw conversation session",
                    "kind": "session",
                    "score": raw_session.score,
                    "citations": [{"event_id": str(event_id), "excerpt": raw_session.text[:500]} for event_id in raw_session.event_ids[:1]],
                    "evidence_sessions": [raw_session.session_id],
                    "primary_session": raw_session.session_id,
                    "documentDate": raw_session.occurred_at,
                    "eventDate": [raw_session.occurred_at] if raw_session.occurred_at else [],
                    "chunks": raw_session.text,
                }
            )
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

        # Retrieval quality: use search results directly (no token budget / prompt wrappers)
        retrieval_limit = _get_retrieval_limit(args)
        packed_parts = [
            f"Memory: Raw conversation session\nChunks: {item.text}\ndocumentDate: {item.occurred_at or ''}\neventDate: {item.occurred_at or ''}"
            for item in raw_sessions[:retrieval_limit]
        ]
        packed_parts.extend(f"Memory: {hit.statement}\nChunks: {hit.evidence_excerpt or ''}" for hit in ranked[:retrieval_limit])
        packed_text = "" if abstained else "\n\n".join(packed_parts[:retrieval_limit])

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
            {
                "memory_id": str(m.memory_id),
                "statement": m.statement,
                "kind": m.kind,
                "status": m.status,
                "citations": [{"event_id": str(c.event_id)} for c in m.citations],
            }
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
            "candidate_count": len(search_results) + len(raw_sessions),
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
        if getattr(args, "baseline", "system") == "full-history":
            order = [session_id for session_id, _, _ in sample.sessions]
            return {"session_order": order, "best_rank": next((i for i, sid in enumerate(order[:args.recall_k], 1) if sid in sample.answer_session_ids), None), "ndcg": 1.0 if sample.answer_session_ids else 0.0, "abstained": False, "packed": "\n".join(t["content"] for _, _, turns in sample.sessions for t in turns), "packed_words": sample.raw_words, "latency_ms": 0.0, "candidate_count": len(order)}
        limit = max(args.recall_k * 10, 50)
        ns = sample.question_id if getattr(args, "single_db", False) else None
        if args.no_dense:
            hits = search_atoms(db, sample.question, limit, vector_search=lambda *_: [], namespace_id=ns)
        else:
            hits = search_atoms(db, sample.question, limit, vector_search=lambda query, lim: shared_dense(db, query, lim, namespace_id=ns), namespace_id=ns)
        ranked = hits
        abstained = False
        if not args.no_rerank:
            reranked = rerank_hits(sample.question, hits, args.abstain_threshold)
            abstained = reranked is None
            ranked = reranked if reranked is not None else []
        latency_ms = (time.perf_counter() - started) * 1000
        session_order: list[str] = []
        seen: set[str] = set()
        for hit in ranked:
            if hit.session_id not in seen:
                seen.add(hit.session_id)
                session_order.append(str(hit.session_id))
        retrieval_limit = _get_retrieval_limit(args)
        packed = "" if abstained else "\n".join(hit.fact for hit in ranked[:retrieval_limit])
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


ANSWER_SYSTEM = """You are a question-answering system. Answer only from Retrieved Context.

Each result may contain Memory (a summary), Chunks (the raw source text),
documentDate (when it was written), and eventDate (when it happened). Read
chunks for detail, use dates to resolve time, and combine results only when
the context supports it. Give a clear, concise answer. If the context lacks
the answer, reply exactly: insufficient information."""

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
                {
                    "role": "user",
                    "content": (
                        f"Question: {sample.question}\n"
                        f"Question Date: {sample.question_date}\n\n"
                        f"Retrieved Context:\n{context}\n\nAnswer:"
                    ),
                },
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
    ingest_target = sample
    if getattr(args, "baseline", "system") == "oracle":
        ingest_target = replace(sample, sessions=tuple(s for s in sample.sessions if s[0] in sample.answer_session_ids))
    database_path = ingest_sample(Path(args.work_dir), ingest_target, skip_embeddings=args.no_dense, single_db=getattr(args, "single_db", False))
    outcome = retrieve_session_ranking(database_path, sample, args)
    trace: dict[str, Any] = {
        "status": "completed",
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "answer_session_ids": sorted(sample.answer_session_ids),
        "best_rank": outcome["best_rank"],
        "ndcg_at_k": round(outcome["ndcg"], 4),
        "abstained": outcome["abstained"],
        "unanswerable": sample.unanswerable or sample.question_id.endswith("_abs"),
        "abstention_correct": bool(outcome["abstained"] == (sample.unanswerable or sample.question_id.endswith("_abs"))),
        "recall": {str(k): int(outcome["best_rank"] is not None and outcome["best_rank"] <= k) for k in (5, 10, args.recall_k)},
        "packed_words": outcome["packed_words"],
        "raw_words": sample.raw_words,
        "retrieval_latency_ms": outcome["latency_ms"],
        "candidate_count": outcome["candidate_count"],
        "retrieved_memory_count": outcome["candidate_count"],
        "session_order": outcome["session_order"],
    }
    if (args.mode == "judged" or getattr(args, "judge", False)) and budget is not None:
        judged = judge_question(args.answer_model, args.judge_model, sample, outcome["packed"], budget)
        trace.update(judged)
    return trace


def evaluate_sample_e2e(args: argparse.Namespace, sample: Sample, budget: OpenRouterBudget | None) -> dict[str, Any]:
    """Evaluate one sample through production pipeline.

    Steps (with leakage boundary):
      1. ingest_e2e  uses ONLY haystack_sessions (no question/answer)
      2. retrieve_e2e_session_ranking  uses question ONLY for retrieval
    """
    work_dir = Path(args.work_dir)
    ingest_start = time.perf_counter()
    db_path, e2e_diag = ingest_e2e(work_dir, sample, args)
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
        # Check ranking miss (no token-budget after context removal)
        if retrieval_outcome.get("abstained"):
            failure_reason = "abstained"
        else:
            failure_reason = "ranking_miss"

    # Build oracle session texts for failure analysis (without leaking into extraction)
    oracle_texts = []
    for sid, _, turns in sample.sessions:
        if sid in sample.answer_session_ids:
            oracle_texts.append({"session_id": sid, "turns": [{"role": t["role"], "content": t["content"][:500]} for t in turns]})

    trace: dict[str, Any] = {
        "status": "completed",
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "question": sample.question,
        "answer": sample.answer,
        "answer_session_ids": sorted(sample.answer_session_ids),
        "oracle_session_texts": oracle_texts,
        "best_rank": best_rank,
        "ndcg_at_k": round(retrieval_outcome["ndcg"], 4),
        "abstained": retrieval_outcome["abstained"],
        "recall": {str(k): int(best_rank is not None and best_rank <= k) for k in (5, 10, args.recall_k)},
        "packed_words": retrieval_outcome["packed_words"],
        "raw_words": sample.raw_words,
        "retrieval_latency_ms": retrieval_outcome["latency_ms"],
        "candidate_count": retrieval_outcome["candidate_count"],
        "retrieved_memory_count": retrieval_outcome["candidate_count"],
        # Memory-formation diagnostics
        "e2e_diagnostics": e2e_diag,
        "events_ingested": e2e_diag.get("events_ingested", 0),
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
    if (args.mode == "judged" or getattr(args, "judge", False)) and budget is not None:
        judged = judge_question(args.answer_model, args.judge_model, sample, retrieval_outcome["packed"], budget)
        trace.update(judged)
    return trace


def summarize(traces: list[dict[str, Any]], recall_k: int, judged: bool) -> list[dict[str, Any]]:
    successful = [trace for trace in traces if trace.get("status", "completed") == "completed"]
    ks = ["5", "10", str(recall_k)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in successful:
        grouped[trace["question_type"]].append(trace)
    rows: list[dict[str, Any]] = []

    def row_for(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        row: dict[str, Any] = {"Category": name, "Completed": count}
        scored = [item for item in items if not item.get("unanswerable", False)]
        for k in ks:
            row[f"Recall@{k} (%)"] = round(100 * sum(item["recall"][k] for item in scored) / len(scored), 1) if scored else None
        row[f"MRR@{recall_k}"] = round(sum((1 / item["best_rank"]) if item["best_rank"] else 0.0 for item in scored) / len(scored), 3) if scored else None
        row[f"NDCG@{recall_k}"] = round(sum(item["ndcg_at_k"] for item in scored) / len(scored), 3) if scored else None
        row["Avg Retrieved Words"] = round(sum(item["packed_words"] for item in items) / count, 1) if count else None
        row["Avg Latency (ms)"] = round(sum(item["retrieval_latency_ms"] for item in items) / count, 1) if count else None
        if count:
            row["Abstention Rate (%)"] = round(100 * sum(bool(item.get("abstained")) for item in items) / count, 1)
            unanswerable = [item for item in items if item.get("unanswerable")]
            row["Abstention Accuracy (%)"] = round(100 * sum(bool(item.get("abstention_correct")) for item in unanswerable) / len(unanswerable), 1) if unanswerable else None
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
    rows.append(row_for("Overall", successful))
    return rows


def failure_decomposition(traces: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [trace for trace in traces if trace.get("status", "completed") == "completed"]
    counter = Counter(t.get("failure_reason", "unknown") for t in completed if t.get("best_rank") is None)
    return {"counts": dict(counter), "total_missed": sum(counter.values()), "completed": len(completed)}


def failed_trace(sample: Sample, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "error_class": type(exc).__name__,
        "error": str(exc),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return None


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("N/A" if row.get(header) is None else str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    global _openrouter_pacer

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    # --micro convenience: use the 30-sample stratified subset (5 per category)
    if getattr(args, "micro", False):
        # allow --micro to override default data path unless user explicitly set --data-path
        # Detect if data_path is still default; if so swap to micro path
        if Path(args.data_path) == DEFAULT_DATA_PATH:
            args.data_path = DEFAULT_MICRO_PATH
            print(f"Using micro dataset: {args.data_path} (30 samples, 5 per category)", flush=True)
        else:
            print(f"--micro set but --data-path already overridden to {args.data_path}; keeping explicit path", flush=True)
    data_path = Path(args.data_path)
    dataset_bytes = data_path.read_bytes()
    dataset_sha256 = hashlib_sha256(dataset_bytes)
    samples = normalize_samples(json.loads(dataset_bytes.decode("utf-8")))
    if data_path.name == DEFAULT_DATA_PATH.name and len(samples) != 500:
        raise SystemExit(f"standard LongMemEval-S dataset must contain 500 samples (found {len(samples)})")
    if data_path.name == DEFAULT_MICRO_PATH.name:
        counts = Counter(sample.question_type for sample in samples)
        if len(samples) != 30 or any(counts.get(category, 0) != 5 for category in CATEGORY_ORDER):
            raise SystemExit("micro dataset must contain 30 samples with five per category")
    if args.task:
        samples = [item for item in samples if item.question_type == args.task]
    resume_ids: set[str] = set()
    previous_traces: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    if args.resume_from:
        previous = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
    elif getattr(args, "resume_work_dir", None):
        checkpoint_file = _checkpoint_path(Path(args.resume_work_dir))
        if not checkpoint_file.exists():
            raise SystemExit(f"no checkpoint found in {Path(args.resume_work_dir)}")
        previous = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    if previous is not None:
        previous_traces = [trace for trace in previous.get("traces", []) if trace.get("status", "completed") == "completed"]
        resume_ids = {trace["question_id"] for trace in previous_traces if trace.get("status", "completed") == "completed"}
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
        _openrouter_pacer = RequestPacer(float(getattr(args, "openrouter_min_interval", 3.0)))
        try:
            work_dir = resolve_work_dir(args, previous, is_e2e=True)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.work_dir = str(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Work directory: {work_dir}", flush=True)
    log_path = Path(args.log_file) if getattr(args, "log_file", None) else work_dir / "benchmark.log"
    _configure_benchmark_logging(log_path, append=previous is not None)
    print(f"Detailed log: {log_path}", flush=True)
    should_judge = canonical_mode == "judged" or bool(getattr(args, "judge", False))
    budget = OpenRouterBudget(args.budget_usd) if should_judge else None

    manifest = _run_manifest(args, data_path, dataset_sha256, canonical_mode)
    manifest_path = _run_manifest_path(work_dir)
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _normalize_manifest_for_comparison(previous_manifest) != _normalize_manifest_for_comparison(manifest):
            raise SystemExit(
                f"work dir {work_dir} already has a different LongMemEval run manifest; use a fresh work dir or pass --resume-from for the same run"
            )
    # Write canonical manifest (migrates legacy work dirs: pack_atoms/token_budget -> retrieval_limit)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    traces: list[dict[str, Any]] = list(previous_traces)
    failures = sum(trace.get("status") == "failed" for trace in previous_traces)
    started = time.perf_counter()

    def worker_retrieval(sample: Sample) -> dict[str, Any]:
        return evaluate_sample(args, sample, budget)

    def worker_e2e(sample: Sample) -> dict[str, Any]:
        return evaluate_sample_e2e(args, sample, budget)

    worker = worker_e2e if is_e2e else worker_retrieval

    total_questions = len(traces) + len(pending)

    def print_progress(sample: Sample, *, failed: bool = False, error: Exception | None = None) -> None:
        finished = len(traces)
        percent = (100.0 * finished / total_questions) if total_questions else 100.0
        answered = sum("hypothesis" in item for item in traces if item.get("status") == "completed")
        failed_count = sum(item.get("status") == "failed" for item in traces)
        answer_label = f"answered: {answered}" if should_judge else f"completed: {finished - failed_count}"
        suffix = f" | failed: {failed_count}" if failed_count else ""
        if failed and error is not None:
            suffix += f" | last error: {type(error).__name__}: {error}"
        print(f"Progress: {finished}/{total_questions} ({percent:.0f}%) | {answer_label}{suffix}", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {pool.submit(worker, sample): sample for sample in pending}
        for number, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            try:
                trace = future.result()
                traces.append(trace)
                _write_checkpoint(work_dir, manifest=manifest, args=args, dataset_sha256=dataset_sha256, traces=traces)
                _BENCHMARK_LOGGER.info("completed question=%s category=%s rank=%s", sample.question_id, sample.question_type, trace["best_rank"])
                print_progress(sample)
            except BudgetExceeded as exc:
                failures += 1
                traces.append(failed_trace(sample, exc))
                _write_checkpoint(work_dir, manifest=manifest, args=args, dataset_sha256=dataset_sha256, traces=traces)
                _BENCHMARK_LOGGER.exception("budget stop for question=%s", sample.question_id)
                print(f"BUDGET STOP: {exc}. See {log_path}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as exc:
                failures += 1
                traces.append(failed_trace(sample, exc))
                _write_checkpoint(work_dir, manifest=manifest, args=args, dataset_sha256=dataset_sha256, traces=traces)
                _BENCHMARK_LOGGER.exception("failed question=%s category=%s", sample.question_id, sample.question_type)
                print_progress(sample, failed=True, error=exc)
    except KeyboardInterrupt:
        _write_checkpoint(work_dir, manifest=manifest, args=args, dataset_sha256=dataset_sha256, traces=traces)
        pool.shutdown(wait=False, cancel_futures=True)
        _BENCHMARK_LOGGER.info("interrupted; checkpoint=%s", _checkpoint_path(work_dir))
        print(f"\nInterrupted. Checkpoint saved: {_checkpoint_path(work_dir)}", flush=True)
        print(f"Resume with: --resume-work-dir {work_dir}", flush=True)
        return 130
    else:
        pool.shutdown(wait=True)

    rows = summarize(traces, args.recall_k, judged=should_judge)
    table = render_table(rows)
    git_commit = _git_commit()
    # Embedding provider info
    embed_name = shared_product_embedder(args).name if is_e2e else shared_embedder().name
    # Extraction provider info
    extraction_provider = getattr(args, "extraction", "openrouter") if is_e2e else "verbatim-atoms"
    extraction_model = getattr(args, "extraction_model", None)
    completed = sum(trace.get("status", "completed") == "completed" for trace in traces)
    result = {
        "mode": canonical_mode,
        "raw_mode": raw_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "dataset": {"path": str(data_path), "sha256": dataset_sha256, "samples_total": len(samples)},
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "samples_completed": completed,
        "samples_failed": failures,
        "samples_total": len(samples),
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
            "question_count": completed,
            "extraction_provider": extraction_provider,
            "extraction_model": str(extraction_model) if extraction_model else None,
            "embedding_provider": embed_name,
            "reranker": "ms-marco-MiniLM-L-12-v2" if not args.no_rerank else None,
            "dense_enabled": not args.no_dense,
            "workers": args.workers,
            "retrieval_limit": _get_retrieval_limit(args),
            "top_k_values": [5, 10, args.recall_k],
            "recall_k": args.recall_k,
            "abstain_threshold": args.abstain_threshold,
        },
    }
    output_path = Path(args.results_dir) / f"longmemeval_s_{canonical_mode}_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_checkpoint(work_dir, manifest=manifest, args=args, dataset_sha256=dataset_sha256, traces=traces)
    print("\n" + table)
    if is_e2e and result.get("failure_decomposition"):
        print("\nFailure decomposition:", json.dumps(result["failure_decomposition"], indent=2))
    print(f"\nTraces: {output_path}")
    if budget:
        print(f"Spend: ${budget.spent_usd:.4f}")
    return 1 if failures else 0


def hashlib_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="TermyteDB LongMemEval-S benchmark")
    parser.add_argument("--baseline", choices=("system", "oracle", "full-history"), default="system", help="evaluation baseline")
    parser.add_argument(
        "--mode",
        choices=("retrieval", "retrieval-only", "end-to-end", "judged", "end_to_end", "e2e"),
        default="end-to-end",
        help="Benchmark pipeline: retrieval-only (verbatim atoms) or end-to-end (production events)",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--micro",
        action="store_true",
        help="use longmemeval-micro (30 samples, 5 per category) instead of full 500; ~94%% cheaper/faster",
    )
    parser.add_argument("--work-dir", default=str(ROOT / ".termytedb-work" / "longmemeval"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--log-file", type=Path, help="path for detailed benchmark and engine logs (default: <work-dir>/benchmark.log)")
    parser.add_argument("--limit", type=int, help="limit number of questions (for smoke tests)")
    parser.add_argument("--smoke", action="store_true", help="run the manual 5-sample benchmark smoke subset")
    parser.add_argument("--smoke-samples", type=int, default=5, help="number of questions to use for smoke runs")
    parser.add_argument("--confirm-benchmark", action="store_true", help="required to start a benchmark smoke loop")
    parser.add_argument("--task", choices=CATEGORY_ORDER, help="filter to single question_type")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recall-k", type=int, default=15)
    parser.add_argument("--retrieval-limit", type=int, default=40, help="max memories/atoms to retrieve per query (replaces --pack-atoms)")
    parser.add_argument("--pack-atoms", type=int, default=None, help="deprecated alias for --retrieval-limit")
    parser.add_argument("--abstain-threshold", type=float, default=0.25)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--single-db", action="store_true", help="store all questions in one SQLite file (namespace-isolated)")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-work-dir", type=Path, help="resume an interrupted end-to-end run from its work-directory checkpoint")
    parser.add_argument("--judge", action="store_true", help="run answer generation and judging after end-to-end retrieval")
    parser.add_argument("--answer-model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--budget-usd", type=float, default=8.0)
    parser.add_argument("--embedding-provider", choices=("local", "openrouter"), default=None, help="embedding provider for end-to-end runs")
    parser.add_argument("--embedding-model", type=str, default=None, help="model for OpenRouter-compatible embeddings")
    parser.add_argument("--embedding-dimensions", type=int, default=None, help="dimensions for OpenRouter-compatible embeddings")
    # End-to-end extraction config
    parser.add_argument(
        "--extraction",
        choices=("openrouter", "fake", "http"),
        default="openrouter",
        help="extraction provider for end-to-end mode (OpenRouter is the product default)",
    )
    parser.add_argument("--extraction-model", type=str, default=None, help="model for openrouter/http extraction (or env TERMYTEDB_EXTRACTION_MODEL)")
    parser.add_argument("--extraction-batch-sessions", type=int, default=4, help="complete sessions sent in one extraction call (default: 4)")
    parser.add_argument("--openrouter-min-interval", type=float, default=3.0, help="minimum seconds between all OpenRouter requests")
    parser.add_argument("--openrouter-max-retries", type=int, default=5, help="maximum attempts for retryable OpenRouter extraction failures")
    parser.add_argument("--openrouter-rate-limit-cooldown", type=float, default=60.0, help="seconds to wait after an OpenRouter HTTP 429")
    args = parser.parse_args()
    # Deprecated alias handling: --pack-atoms -> --retrieval-limit
    if getattr(args, "pack_atoms", None) is not None:
        import warnings

        warnings.warn("--pack-atoms is deprecated; use --retrieval-limit", DeprecationWarning, stacklevel=2)
        if getattr(args, "retrieval_limit", 40) == 40:  # only override if retrieval_limit is default
            args.retrieval_limit = args.pack_atoms
    elif getattr(args, "pack_atoms", None) is None:
        args.pack_atoms = getattr(args, "retrieval_limit", 40)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

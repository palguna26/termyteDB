"""LongMemEval-S benchmark for TermyteDB.

Canonical harness. Feeds the same episodic retrieval stack the engine ships:
verbatim turn-level atoms -> FTS5 + dense hybrid (RRF) -> FlashRank rerank ->
session aggregation -> bounded context packing.

Modes:
  retrieval  Zero-cost session-level retrieval metrics (Recall@k, MRR, NDCG).
  judged     Adds answer generation + LLM judging through OpenRouter with a
             hard spend budget.

Methodology notes are in docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
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


class GuardedEmbedder:
    """Serializes ONNX inference so concurrent workers cannot grow competing arenas."""

    def __init__(self, inner: FastEmbedProvider) -> None:
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


def verbatim_atoms(sample: Sample) -> list[L1Atom]:
    """One atom per message: lossless episodic encoding at zero API cost."""
    atoms: list[L1Atom] = []
    for session_id, date, turns in sample.sessions:
        for turn in turns:
            fact = turn["content"][:MAX_ATOM_CHARS]
            atoms.append(
                L1Atom(atom_id=str(uuid4()), session_id=session_id, fact=fact, timestamp=date or None, source_role=turn["role"])
            )
    return atoms


def ingest_sample(
    work_dir: Path, sample: Sample, *, skip_embeddings: bool = False
) -> Path:
    database_path = work_dir / f"{sample.question_id}.sqlite"
    db = Database(database_path)
    try:
        existing = db.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
        if existing == 0:
            insert_atoms(db, verbatim_atoms(sample))
            if not skip_embeddings:
                index_atom_embeddings(db, shared_embedder(), batch_size=64)
        elif not skip_embeddings:
            # Backfill embeddings for DBs created in --no-dense mode
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


def shared_dense(db: Database, query: str, limit: int) -> list[AtomHit]:
    return dense_search_atoms(db, query, limit, provider=shared_embedder())


def retrieve_session_ranking(database_path: Path, sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    db = Database(database_path)
    started = time.perf_counter()
    try:
        limit = max(args.recall_k * 4, 20)
        if args.no_dense:
            hits = search_atoms(db, sample.question, limit, vector_search=lambda *_: [])
        else:
            hits = search_atoms(db, sample.question, limit, vector_search=lambda query, lim: shared_dense(db, query, lim))
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
        Path(args.work_dir), sample, skip_embeddings=args.no_dense
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
        if judged:
            row["Judged Acc (%)"] = round(100 * sum(int(bool(item.get("correct"))) for item in items) / count, 1) if count else 0.0
        return row

    for category in CATEGORY_ORDER:
        if category in grouped:
            rows.append(row_for(category, grouped[category]))
    rows.append(row_for("Overall", traces))
    return rows


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

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    budget = OpenRouterBudget(args.budget_usd) if args.mode == "judged" else None

    traces: list[dict[str, Any]] = list(previous_traces)
    failures = 0
    started = time.perf_counter()

    def worker(sample: Sample) -> dict[str, Any]:
        return evaluate_sample(args, sample, budget)

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
                print(f"[{number}/{len(pending)}] {sample.question_id} ({sample.question_type}): {status}; {trace['retrieval_latency_ms']:.0f}ms", flush=True)
            except BudgetExceeded as exc:
                failures += 1
                print(f"BUDGET STOP: {exc}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as exc:
                failures += 1
                print(f"[{number}/{len(pending)}] {sample.question_id}: FAILED {type(exc).__name__}: {exc}", flush=True)

    rows = summarize(traces, args.recall_k, judged=args.mode == "judged")
    table = render_table(rows)
    result = {
        "mode": args.mode,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {"path": str(data_path), "sha256": dataset_sha256, "samples_total": len(samples)},
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "failures": failures,
        "budget_spent_usd": round(budget.spent_usd, 4) if budget else 0.0,
        "summary": rows,
        "traces": traces,
    }
    output_path = Path(args.results_dir) / f"longmemeval_s_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nTraces: {output_path}")
    if budget:
        print(f"Spend: ${budget.spent_usd:.4f}")
    return 0


def hashlib_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="TermyteDB LongMemEval-S benchmark")
    parser.add_argument("--mode", choices=("retrieval", "judged"), default="retrieval")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--work-dir", default=str(ROOT / ".termytedb-work" / "longmemeval"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", choices=CATEGORY_ORDER)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recall-k", type=int, default=15)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--pack-atoms", type=int, default=40)
    parser.add_argument("--abstain-threshold", type=float, default=0.25)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--answer-model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--budget-usd", type=float, default=8.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

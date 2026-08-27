"""Full TermyteDB E2E LongMemEval-S — full pipeline, zero-config.

Single DB, single namespace (longmemeval-e2e), unique sessions deduped.

Defaults (no flags needed): 500 questions, Ling 3.0 Flash for extraction+answer,
GPT-4o-mini only for judging, API key + models from .env, no budget cap.

Extraction:  inclusionai/ling-3.0-flash via OpenRouter (TERMYTEDB_EXTRACTION_MODEL)
Answer:      inclusionai/ling-3.0-flash via OpenRouter (TERMYTEDB_ANSWER_MODEL)
Judge:       openai/gpt-4o-mini via OpenRouter (TERMYTEDB_JUDGE_MODEL — only judge touches this)
Embeddings:  local FastEmbed bge-small (free)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from termytedb import TermyteDB
from termytedb.api.schemas import ExtractionRequest
from termytedb.memory.provider import OpenRouterExtractionProvider, ProviderError, ProviderResult
from termytedb.retrieval.embedding import FastEmbedProvider

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BENCHMARK_DIR / "longmemeval_s_cleaned.json"
DEFAULT_EXTRACTION_MODEL = "inclusionai/ling-3.0-flash"
DEFAULT_ANSWER_MODEL = "inclusionai/ling-3.0-flash"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


class PacedExtractionProvider:
    """Wraps OpenRouterExtractionProvider with global pacing to avoid 429 dead-letters.
    Optimized: Ling via Novita tolerates 0.76s P50, so 0.7s global pacing + 8 workers
    cuts 19k sessions from 10.6h -> ~3.7h (was 2.0s -> 10.6h)."""

    name = "openrouter-paced"

    def __init__(self, model: str, api_key: str, min_delay: float = 0.7) -> None:
        self.model = model
        self.api_key = api_key
        self.min_delay = min_delay
        self._lock = threading.Lock()
        self._last = 0.0
        self._inner = OpenRouterExtractionProvider(model=model, api_key=api_key)

    def _pace(self) -> None:
        with self._lock:
            delay = self.min_delay - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()

    def extract(self, request: ExtractionRequest, timeout_seconds: float = 60.0, cancellation=None) -> ProviderResult:
        last_err: ProviderError | None = None
        for attempt in range(4):
            self._pace()
            try:
                return self._inner.extract(request, timeout_seconds=timeout_seconds, cancellation=cancellation)
            except ProviderError as e:
                last_err = e
                # 429 is retryable; back off and retry
                if not e.retryable or "429" not in str(e):
                    # Still retry once for transient 429 with longer backoff
                    if attempt < 3 and e.retryable:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise last_err
        raise ProviderError("extraction failed", retryable=True, error_class="unknown")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    for rank, sid in enumerate(retrieved, 1):
        if sid in expected:
            return 1.0 / rank
    return 0.0


# ---- OpenRouter direct judge (so extraction stays Mistral, judge stays GPT-4o-mini)
import threading
from urllib.request import Request, urlopen

_judge_lock = threading.Lock()
_spent_usd = 0.0
_spent_lock = threading.Lock()

JUDGE_SYSTEM = (
    "You are an evaluator for a conversational memory benchmark. Given a question, a reference answer, "
    "and a system response, decide whether the response conveys the reference answer. "
    'Reply with JSON only: {"correct": true} or {"correct": false}.'
)


def openrouter_chat(model: str, messages: list[dict[str, str]], timeout: float = 60.0) -> tuple[str, float]:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    payload = json.dumps({"model": model, "messages": messages}).encode()
    last_err = None
    for attempt in range(4):
        try:
            req = Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            cost = float(body.get("usage", {}).get("cost", 0.0))
            with _spent_lock:
                global _spent_usd
                _spent_usd += cost
            return content, cost
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt + 0.5, 8))
    raise RuntimeError(f"OpenRouter chat failed: {last_err}")


def judge_via_gpt4o_mini(question: str, answer: str, hypothesis: str, judge_model: str) -> tuple[bool, str]:
    prompt = (
        f"Question: {question}\nReference answer: {answer}\n"
        f"System response: {hypothesis}\n\n"
        "Does the system response convey the reference answer? "
        'Reply JSON only: {"correct": true/false}.'
    )
    raw, _ = openrouter_chat(
        judge_model,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        return bool(json.loads(raw.strip().strip("`").removeprefix("json"))["correct"]), raw
    except Exception:
        return "true" in raw.lower(), raw


def run(
    dataset: Path,
    top_k: int,
    database: Path | None,
    workers: int,
    extraction_model: str,
    answer_model: str,
    judge_model: str,
    judged: bool,
    limit: int | None,
    budget_usd: float | None,
    token_budget: int,
) -> dict[str, Any]:
    load_dotenv(override=True)
    # Models + key from .env — no flags required; .env overrides defaults
    extraction_model = os.environ.get("TERMYTEDB_EXTRACTION_MODEL", extraction_model)
    answer_model = os.environ.get("TERMYTEDB_ANSWER_MODEL", answer_model)
    judge_model = os.environ.get("TERMYTEDB_JUDGE_MODEL", judge_model)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")
    extraction_provider = PacedExtractionProvider(model=extraction_model, api_key=api_key, min_delay=0.7)
    log(f"Extraction: {extraction_model} (paced 0.7s, 8 workers -> ~3.7h for 19k sessions)")
    log(f"Answer: {answer_model}")
    log(f"Judge: {judge_model} (only judge)")
    if budget_usd is not None:
        log(f"Budget guard: ${budget_usd:.2f}")

    questions = json.loads(dataset.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    log(f"Dataset: {len(questions)} questions from {dataset.name}")

    from tempfile import TemporaryDirectory
    import logging

    tmpdir = None
    if database is None:
        tmpdir = TemporaryDirectory(prefix="termytedb-e2e-")
        database = Path(tmpdir.name) / "e2e.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    log(f"Database: {database} (single file, namespace=longmemeval-e2e)")

    embedding = FastEmbedProvider()
    logger = logging.getLogger("termytedb.e2e")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    db = TermyteDB(database, logger=logger, extraction_provider=extraction_provider, embedding_provider=embedding)
    event_sessions: dict[str, str] = {}
    session_payloads: dict[str, list[dict[str, Any]]] = {}
    try:
        for q in questions:
            sids = q.get("haystack_session_ids", [])
            for idx, sess in enumerate(q.get("haystack_sessions", [])):
                if not isinstance(sess, list) or idx >= len(sids):
                    continue
                sid = str(sids[idx])
                session_payloads.setdefault(sid, sess)
        log(f"Unique sessions to ingest: {len(session_payloads)} (deduped from {sum(len(q.get('haystack_sessions', [])) for q in questions)} haystacks)")

        for i, (sid, msgs) in enumerate(session_payloads.items(), 1):
            receipt = db.ingest(
                {
                    "namespace_id": "longmemeval-e2e",
                    "idempotency_key": f"longmemeval:{sid}",
                    "type": "conversation.session",
                    "session_id": sid,
                    "stream_id": sid,
                    "payload": {"messages": msgs},
                }
            )
            event_sessions[str(receipt.event_id)] = sid
            if i == 1 or i % 500 == 0:
                log(f"Ingested {i}/{len(session_payloads)}")

        log(f"Processing with {workers} workers (Ling extraction, ~248 pairs/q avg)")
        processed = 0
        worker_dbs = [TermyteDB(database, logger=logger, extraction_provider=extraction_provider, embedding_provider=embedding) for _ in range(workers)]
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                while True:
                    resps = list(ex.map(lambda w: w.process_with_timeout("longmemeval-e2e", limit=20, timeout_seconds=300.0), worker_dbs))
                    batch = sum(r.processed for r in resps)
                    processed += batch
                    failed = sum(r.failed for r in resps)
                    dead = sum(r.dead_lettered for r in resps)
                    metrics = db.metrics("longmemeval-e2e")
                    pending = int(metrics.get("jobs_pending", 0))
                    processing = int(metrics.get("jobs_processing", 0))
                    # Fallback: count directly if metrics missing
                    if "jobs_processing" not in metrics:
                        processing = int(db.database.execute("SELECT COUNT(*) FROM processing_jobs WHERE namespace_id=? AND status='processing'", ("longmemeval-e2e",)).fetchone()[0])
                    jobs_dead = int(metrics.get("jobs_dead", 0))
                    log(f"Processed {processed} | failed={failed} dead={dead} pending={pending} processing={processing} dead_total={jobs_dead}")
                    if dead and jobs_dead > 20:
                        log(f"WARNING: {jobs_dead} dead-lettered jobs — continuing, recall will be lower")
                    if pending == 0 and processing == 0:
                        break
                    if batch == 0 and failed == 0 and pending == 0 and processing > 0:
                        log(f"Waiting for {processing} leased jobs to complete or expire...")
                        time.sleep(5)
                    elif batch == 0 and failed:
                        time.sleep(3)
        finally:
            for w in worker_dbs:
                w.close()

        log(f"Evaluating {len(questions)} questions @ top_k={top_k}" + (" + GPT-4o-mini judged" if judged else " (retrieval only)"))
        rows: list[dict[str, Any]] = []
        by_type: dict[str, list[bool]] = defaultdict(list)
        judged_correct: list[bool] = []
        query_times: list[float] = []
        ingest_memories = int(db.metrics("longmemeval-e2e").get("memories", 0))
        log(f"Memories materialized: {ingest_memories}")

        for idx, q in enumerate(questions, 1):
            expected = {str(x) for x in q.get("answer_session_ids", [])}
            if not expected:
                continue
            qid = q["question_id"]
            qtype = q["question_type"]
            t0 = time.perf_counter()
            hits = db.search("longmemeval-e2e", str(q["question"]), limit=top_k)
            query_times.append((time.perf_counter() - t0) * 1000)
            retrieved: list[str] = []
            for h in hits:
                for c in h.citations:
                    sid = event_sessions.get(str(c.event_id))
                    if sid and sid not in retrieved:
                        retrieved.append(sid)
            retrieved = retrieved[:top_k]
            hit = bool(expected.intersection(retrieved))
            mrr = reciprocal_rank(retrieved, expected)
            by_type[qtype].append(hit)
            row: dict[str, Any] = {
                "question_id": qid,
                "question_type": qtype,
                "hit": hit,
                "mrr": mrr,
                "retrieved": retrieved,
                "expected": sorted(expected),
            }
            if judged:
                ctx = db.context("longmemeval-e2e", str(q["question"]), token_budget=token_budget, limit=top_k)
                hypothesis, _ = openrouter_chat(
                    answer_model,
                    [
                        {"role": "system", "content": "Answer using ONLY the provided context. If insufficient, say insufficient information."},
                        {"role": "user", "content": f"Context:\n{ctx.text}\n\nQuestion: {q['question']}"},
                    ],
                )
                correct, raw = judge_via_gpt4o_mini(str(q["question"]), str(q["answer"]), hypothesis, judge_model)
                row["judged_correct"] = correct
                row["hypothesis"] = hypothesis[:400]
                row["judge_raw"] = raw[:400]
                judged_correct.append(correct)
                if budget_usd is not None and _spent_usd > budget_usd:
                    log(f"Budget ${budget_usd:.2f} exceeded (${_spent_usd:.4f}); stopping judged loop")
                    rows.append(row)
                    break
            rows.append(row)
            if idx == 1 or idx % 25 == 0:
                extra = f" judged={'yes' if row.get('judged_correct') else 'no'}" if judged and "judged_correct" in row else ""
                log(f"{idx}/{len(questions)} {qid} hit={'yes' if hit else 'no'}{extra}")

        recall = sum(int(r["hit"]) for r in rows) / len(rows) if rows else 0.0
        mrr_avg = sum(float(r["mrr"]) for r in rows) / len(rows) if rows else 0.0
        result: dict[str, Any] = {
            "pipeline": f"TermyteDB ingest -> {extraction_model} extraction + {answer_model} answer -> {judge_model} judge",
            "dataset": dataset.name,
            "extraction_model": extraction_model,
            "answer_model": answer_model,
            "judge_model": judge_model if judged else None,
            "questions": len(rows),
            "top_k": top_k,
            "token_budget": token_budget,
            "unique_sessions": len(session_payloads),
            "processed_jobs": processed,
            "memories": ingest_memories,
            "query_p50_ms": round(statistics.median(query_times), 2) if query_times else 0,
            "query_p95_ms": round(sorted(query_times)[max(0, int(len(query_times) * 0.95) - 1)], 2) if query_times else 0,
            "recall": round(recall, 4),
            "mrr": round(mrr_avg, 4),
            "by_type": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())},
            "judged_accuracy": round(sum(judged_correct) / len(judged_correct), 4) if judged_correct else None,
            "judged_n": len(judged_correct) if judged else 0,
            "spent_usd": round(_spent_usd, 4),
            "misses": [r for r in rows if not r["hit"]][:10],
        }
        log(f"Done: Recall@{top_k}={recall:.3f} MRR={mrr_avg:.3f}" + (f" Judged={result['judged_accuracy']:.3f} ({len(judged_correct)})" if judged_correct else "") + f" spent=${_spent_usd:.4f}")
        return result
    finally:
        db.close()
        if tmpdir is not None:
            tmpdir.cleanup()


def main() -> int:
    p = argparse.ArgumentParser(description="TermyteDB full E2E LongMemEval-S — defaults to 500q, Ling 3.0 Flash extraction+answer, GPT-4o-mini judge (.env)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--database", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extraction-model", default=DEFAULT_EXTRACTION_MODEL)
    p.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judged", action=argparse.BooleanOptionalAction, default=True, help="run answer+judge (default: on; use --no-judged for retrieval-only)")
    p.add_argument("--token-budget", type=int, default=1500)
    p.add_argument("--limit", type=int, default=None, help="limit questions (e.g., 10 for smoke; default: all 500)")
    p.add_argument("--budget-usd", type=float, default=None, help="optional hard stop when spend exceeds this (default: no cap)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    res = run(
        dataset=args.dataset,
        top_k=args.top_k,
        database=args.database,
        workers=args.workers,
        extraction_model=args.extraction_model,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        judged=args.judged,
        limit=args.limit,
        budget_usd=args.budget_usd if args.judged else None,
        token_budget=args.token_budget,
    )
    out = json.dumps(res, indent=2, ensure_ascii=False)
    print(out)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n", encoding="utf-8")
        log(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

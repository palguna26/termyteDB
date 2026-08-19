from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from termytedb.evaluation.longmemeval_extraction import L1Atom, index_atom_embeddings, insert_atoms
from termytedb.retrieval.embedding import FastEmbedProvider, OpenAICompatibleEmbeddingProvider
from termytedb.retrieval.retrieval import dense_search_atoms, search_atoms
from termytedb.storage.db import Database


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    for rank, session_id in enumerate(retrieved, 1):
        if session_id in expected:
            return 1.0 / rank
    return 0.0


def run(path: Path, top_k: int, database_path: Path | None = None, batch_size: int = 64, mode: str = "dense", provider_kind: str = "local") -> dict[str, object]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("expected the official 500-question LongMemEval-S cleaned dataset")
    temporary: TemporaryDirectory[str] | None = None
    if database_path is None:
        temporary = TemporaryDirectory(prefix="termytedb-longmemeval-")
        database_path = Path(temporary.name) / "benchmark.sqlite"
    db = Database(database_path)
    started = time.perf_counter()
    try:
        atoms: list[L1Atom] = []
        for question in questions:
            question_id = str(question["question_id"])
            session_ids = question.get("haystack_session_ids", [])
            dates = question.get("haystack_dates", [])
            for session_index, session in enumerate(question.get("haystack_sessions", [])):
                session_id = str(session_ids[session_index])
                timestamp = str(dates[session_index]) if session_index < len(dates) else None
                content = "\n".join(str(turn.get("content", "")) for turn in session if isinstance(turn, dict) and turn.get("content"))[:1200]
                if content:
                    atoms.append(L1Atom(f"{question_id}:{session_id}", session_id, content, timestamp, "user"))
        inserted = insert_atoms(db, atoms)
        provider = (OpenAICompatibleEmbeddingProvider() if provider_kind == "openrouter" else FastEmbedProvider()) if mode == "dense" else None
        indexed = index_atom_embeddings(db, provider, batch_size=batch_size) if provider is not None else 0
        index_ms = (time.perf_counter() - started) * 1000
        rows: list[dict[str, object]] = []
        by_type: dict[str, list[bool]] = defaultdict(list)
        query_times: list[float] = []
        for question in questions:
            expected = {str(item) for item in question.get("answer_session_ids", [])}
            if not expected:
                continue
            query_started = time.perf_counter()
            vector_search = (lambda query, limit: dense_search_atoms(db, query, limit, provider)) if provider is not None else None
            hits = search_atoms(db, str(question["question"]), limit=max(top_k * 8, 40), vector_search=vector_search)
            query_times.append((time.perf_counter() - query_started) * 1000)
            session_scores: dict[str, float] = {}
            for hit in hits:
                session_scores[hit.session_id] = max(session_scores.get(hit.session_id, 0.0), hit.score)
            retrieved = [session_id for session_id, _ in sorted(session_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
            hit = bool(expected.intersection(retrieved))
            question_type = str(question.get("question_type", "unknown"))
            by_type[question_type].append(hit)
            rows.append({"question_id": question["question_id"], "question_type": question_type, "retrieved": retrieved, "expected": sorted(expected), "hit": hit, "mrr": reciprocal_rank(retrieved, expected)})
        result = {
            "dataset": path.name,
            "questions": len(rows),
            "top_k": top_k,
            "inserted_turns": inserted,
            "indexed_turns": indexed,
            "index_ms": round(index_ms, 2),
            "query_p50_ms": round(statistics.median(query_times), 2),
            "query_p95_ms": round(sorted(query_times)[max(0, int(len(query_times) * 0.95) - 1)], 2),
            "recall": round(sum(int(row["hit"]) for row in rows) / len(rows), 4),
            "mrr": round(sum(float(row["mrr"]) for row in rows) / len(rows), 4),
            "by_type": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_type.items())},
            "misses": [row for row in rows if not row["hit"]],
        }
        return result
    finally:
        db.close()
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mode", choices=("dense", "fts"), default="dense")
    parser.add_argument("--provider", choices=("local", "openrouter"), default="local")
    args = parser.parse_args()
    database = args.database or (Path("benchmarks/longmemeval") / f"{args.provider}.sqlite")
    result = run(args.dataset, args.top_k, database, args.batch_size, args.mode, args.provider)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

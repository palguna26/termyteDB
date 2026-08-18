from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

from ..retrieval.retrieval import pack_context, rerank_and_filter, search_atoms  # noqa: E402
from ..storage.db import Database  # noqa: E402
from .longmemeval_extraction import L1Atom, index_atom_embeddings, insert_atoms  # noqa: E402

CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    return int(getattr(usage, "prompt_token_count", 0) or 0), int(getattr(usage, "candidates_token_count", 0) or 0)


def _text(response: Any) -> str:
    return str(getattr(response, "text", "") or "").strip()


def _json_text(value: str) -> dict[str, Any]:
    value = value.strip().removeprefix("```json").removesuffix("```").strip()
    return cast(dict[str, Any], json.loads(value))


def _messages(session: Any) -> list[dict[str, Any]]:
    if isinstance(session, list):
        return [item for item in session if isinstance(item, dict)]
    return [{"role": "user", "content": str(session)}]


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * 0.10 / 1_000_000 + output_tokens * 0.40 / 1_000_000


def call_gemini_with_fallback(client: Any, prompt: str, retries: int = 3) -> tuple[Any, str]:
    last_error: Exception | None = None
    for model_name in CANDIDATE_MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                print(f"Connected model: {model_name}")
                return response, model_name
            except Exception as error:
                last_error = error
                if "503" in str(error) and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"[Warning] Endpoint {model_name} failed: {error}. Trying next endpoint...")
                break
    raise RuntimeError("All configured Gemini models failed") from last_error


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LongMemEval file must contain a JSON array")
    return data if limit is None else data[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional bounded run; omit for the full dataset")
    parser.add_argument("--output", type=Path, default=Path("hypothesis_full.jsonl"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY was not loaded from environment")
    if not args.data_path.exists():
        raise SystemExit(f"Data file not found: {args.data_path}")

    from google import genai

    questions = load_questions(args.data_path, args.limit)
    if not questions:
        raise SystemExit("Dataset contains no questions")
    client = genai.Client(api_key=key)
    db = Database(Path("termyte_dryrun.sqlite"))
    ingestion_in = ingestion_out = generation_in = generation_out = 0
    connected_models: list[str] = []
    failed: list[str] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed_ids.add(str(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Detected {len(completed_ids)} completed questions in {args.output}. Resuming remaining items...")
    remaining_questions = [question for question in questions if str(question.get("question_id")) not in completed_ids]
    print(f"Remaining questions: {len(remaining_questions)}")
    output_count = len(completed_ids)
    output_handle = args.output.open("a", encoding="utf-8")
    try:
        for question in remaining_questions:
            question_id = str(question["question_id"])
            if question_id in completed_ids:
                continue
            try:
                session_text = json.dumps(question.get("haystack_sessions", []), ensure_ascii=False)
                extraction_prompt = (
                    "Extract durable facts from these LongMemEval sessions. Return JSON only with an atoms array. "
                    "Each atom must have fact, timestamp (ISO-8601 or null), source_role (user or assistant), and session_id. "
                    "Use one independent third-person fact per atom. Do not infer facts.\n" + session_text
                )
                extraction, extraction_model = call_gemini_with_fallback(client, extraction_prompt)
                connected_models.append(extraction_model)
                used_in, used_out = _usage(extraction)
                ingestion_in += used_in
                ingestion_out += used_out
                parsed = _json_text(_text(extraction))
                atoms = [
                    L1Atom(
                        str(item.get("atom_id") or f"{question_id}-{index}"),
                        str(item.get("session_id") or "unknown"),
                        str(item["fact"]),
                        item.get("timestamp"),
                        str(item.get("source_role", "user")),
                    )
                    for index, item in enumerate(parsed.get("atoms", []))
                    if item.get("fact")
                ]
                insert_atoms(db, atoms)
                try:
                    from ..retrieval.embedding import FastEmbedProvider
                    index_atom_embeddings(db, FastEmbedProvider())
                except ImportError:
                    pass
                hits = search_atoms(db, str(question["question"]), limit=20)
                reranked = rerank_and_filter(str(question["question"]), hits)
                if reranked is None:
                    hypothesis = "Based on the provided evidence, there is no mention of the requested information."
                else:
                    context = pack_context(reranked, token_budget=3000)
                    answer_prompt = (
                        "Answer concisely and strictly using ONLY the provided conversation history. "
                        "If the evidence does not explicitly contain the answer, reply: "
                        "Based on the provided evidence, there is no mention of the requested information.\n"
                        f"CONTEXT:\n{context}\nQUESTION: {question['question']}"
                    )
                    answer, answer_model = call_gemini_with_fallback(client, answer_prompt)
                    connected_models.append(answer_model)
                    used_in, used_out = _usage(answer)
                    generation_in += used_in
                    generation_out += used_out
                    hypothesis = _text(answer)
                output_handle.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed_ids.add(question_id)
                output_count += 1
                print(f"Completed {output_count}/{len(questions)}: {question_id}", flush=True)
            except Exception as error:
                failed.append(question_id)
                print(f"[Failed] {question_id}: {error}", flush=True)
    finally:
        output_handle.close()
        db.close()
    ingestion_usd = _cost_usd(ingestion_in, ingestion_out)
    generation_usd = _cost_usd(generation_in, generation_out)
    total_usd = ingestion_usd + generation_usd
    total_inr = total_usd * float(os.getenv("USD_TO_INR", "85"))
    print("GEMINI_API_KEY loaded successfully: yes")
    print(f"Ingestion tokens: input={ingestion_in}, output={ingestion_out}")
    print("Retrieval tokens: input=0, output=0 (local SQLite/FTS5)")
    print(f"Generation tokens: input={generation_in}, output={generation_out}")
    print(f"Estimated cost: ${total_usd:.8f} / INR {total_inr:.6f}")
    print(f"Processed: {output_count}/{len(questions)}")
    print(f"Failed IDs: {failed}")
    print(f"Wrote {output_count} hypotheses to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

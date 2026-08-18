from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CANDIDATE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"]


def call_judge(client: Any, prompt: str) -> tuple[Any, str]:
    last: Exception | None = None
    for model in CANDIDATE_MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=model, contents=prompt), model
            except Exception as error:
                last = error
                if "503" in str(error) and attempt < 2:
                    time.sleep(2**attempt)
                else:
                    print(f"[Warning] Judge endpoint {model} failed: {error}", flush=True)
                    break
    raise RuntimeError("All Gemini judge endpoints failed") from last


def main() -> int:
    import argparse

    from google import genai

    parser = argparse.ArgumentParser()
    parser.add_argument("hypotheses", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")
    refs = {item["question_id"]: item for item in json.loads(args.references.read_text(encoding="utf-8"))}
    entries = [json.loads(line) for line in args.hypotheses.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = args.output or Path(str(args.hypotheses) + ".gemini-eval.jsonl")
    client = genai.Client(api_key=key)
    total_in = total_out = correct = 0
    model_used = ""
    with output.open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(entries, 1):
            ref = refs.get(entry["question_id"])
            if not ref:
                continue
            prompt = (
                "Answer yes or no only. Decide whether the model response correctly answers the question. "
                "Accept equivalent wording and all required intermediate details. "
                f"\nQuestion: {ref['question']}\nCorrect Answer: {ref['answer']}\n"
                f"Model Response: {entry['hypothesis']}\nIs it correct?"
            )
            response, model_used = call_judge(client, prompt)
            usage = getattr(response, "usage_metadata", None)
            total_in += int(getattr(usage, "prompt_token_count", 0) or 0)
            total_out += int(getattr(usage, "candidates_token_count", 0) or 0)
            label = "yes" in str(getattr(response, "text", "")).lower()
            entry["autoeval_label"] = {"model": model_used, "label": label}
            correct += int(label)
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Judged {index}/{len(entries)}: {entry['question_id']}", flush=True)
    usd = total_in * 0.10 / 1_000_000 + total_out * 0.40 / 1_000_000
    print(f"Scored: {correct}/{len(entries)} ({correct / len(entries):.4f})")
    print(f"Judge tokens: input={total_in}, output={total_out}")
    print(f"Judge cost: ${usd:.8f} / INR {usd * 85:.6f}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

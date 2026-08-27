# TermyteDB LongMemEval-S methodology

This document describes how TermyteDB scores on LongMemEval-S, what the
numbers mean, and how to reproduce them. It exists so a reviewer can rerun the
benchmark and get the same table.

## Dataset

* **Source:** `benchmarks/longmemeval/longmemeval_s_cleaned.json` — mirror of
  `xiaowu0162/LongMemEval` LongMemEval-S split.
* **Size:** 500 questions across 6 types, ~53 haystack sessions per question,
  ~115k tokens of haystack per question.
* **SHA256:** `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
* **Fields per sample:** `question_id`, `question`, `question_type`,
  `question_date`, `haystack_session_ids`, `haystack_dates`,
  `haystack_sessions` (list of `[{role, content}]`), `answer`,
  `answer_session_ids` (oracle session ids that contain the answer).

Distribution in this snapshot: single-session-user 70, single-session-assistant
56, single-session-preference 30, knowledge-update 78, temporal-reasoning 133,
multi-session 133.

## What is measured

The headline benchmark is **end-to-end answer quality and latency**. LongMemEval
questions are pushed through the production memory path:
ingest -> extraction -> evidence validation -> reconciliation -> embeddings ->
retrieval -> context packing -> answer generation -> judging.

The benchmark reports:

* **Answer accuracy** on the judged subset.
* **End-to-end latency** per sample and per stage.
* **Memory formation quality** through accepted/rejected candidate counts,
  evidence validation failures, and retrieval misses.
* **Context size** and packed token usage.

Retrieval-only session recall is kept as an internal ablation and ceiling
measurement. It is not the product claim.

## Engine path under test

Two distinct pipelines are measured, answering different questions:

### A) Retrieval-only ablation
Exercises the product retrieval stack, not a harness-only trick:

1. **Verbatim episodic atoms** — one atom per message (`fact = content`,
   `source_role = role`, `timestamp = haystack_date`). Zero API cost, lossless,
   preserves role and date. Maximum 1500 chars per atom.
2. **Hybrid retrieval** — FTS5 BM25 + dense (FastEmbed `bge-small-en-v1.5`,
   384-d) fused with Reciprocal Rank Fusion (k=60). A `historical` flag gates
   `invalid_at` filtering.
3. **Cross-encoder rerank** — FlashRank `ms-marco-MiniLM-L-12-v2` reranks the
   top 30 candidates (truncated to 600 chars for the cross-encoder; full text
   is kept for packing). Hard abstention when the top score is below
   `--abstain-threshold` (default 0.25).
4. **Session aggregation + packing** — atoms are grouped by `session_id`
   ordered by timestamp; the top `--pack-atoms` atoms are packed into a
   `--token-budget` (default 1500 words) context.

Per-question isolation is enforced with one SQLite database file per
`question_id` under `.termytedb-work/longmemeval/`. No cross-question
contamination. Parallel workers share model instances behind locks to avoid
ONNX arena OOM.

**Use:** ` --mode retrieval-only` (alias `retrieval`). Measures indexing and
retrieval ceiling only.

### B) End-to-end (production pipeline, ordinary agent)
Answers: *If an ordinary agent gave TermyteDB these conversations through the real production interface, would it form the right memories and retrieve them?*

1. **Conversation → EventInput**: `benchmarks/longmemeval/run_benchmark.py:311` `build_event_inputs()` converts `haystack_sessions` into `EventInput` via public `TermyteDB.ingest()` (stream_id=session, actor_id=role, occurred_at from `haystack_dates`, deterministic idempotency `longmemeval:{qid}:{session_index}:{session_id}:{turn}:{hash}`). No `question`/`answer`/`answer_session_ids` leak into this stage.
2. **Processing job → Processor**: `Processor.process_namespace()` claims jobs, runs `payload_text()` → `ExtractionRequest` → `ExtractionProvider` (rule/openrouter/fake) → `validate_candidate()` (evidence offsets + semantic_support) → `reconcile_candidate()` (insert/update/supersede/dispute) → `memory_fts` + `memory_embeddings`.
3. **Retrieval**: same hybrid `Repository.search()` (FTS5 + dense RRF, recency tie-breaker by `valid_from`) + optional FlashRank rerank on `statement` + `build_context(token_budget)`. Session ranking is derived from `evidence_refs → events.stream_id`, not atom `session_id`.
4. **Diagnostics**: per-sample `events_ingested`, `processing_jobs_*`, `candidates_accepted/rejected`, `memories_created`, `rejection_reasons`, `failure_reason` (`never_extracted`, `memory_existed_retrieval_missed`, etc.) and structured `failure_analysis` JSON for automated analysis.

**Use:** ` --mode end-to-end --extraction openrouter --workers 8 --token-budget 1500`.

## Modes and ablations

* `--mode retrieval-only --no-dense`: internal retrieval ceiling, not the headline claim.
* `--mode end-to-end --extraction openrouter --workers 8`: production path with LLM extraction via OpenRouter (`TERMYTEDB_EXTRACTION_MODEL`, `OPENROUTER_API_KEY`).
* `--mode judged`: end-to-end answer generation + OpenRouter judging, budget-guarded.
* `--no-rerank` / `--no-dense --no-rerank`: internal ablations only.
* `--single-db`: single SQLite file with `namespace_id=question_id` isolation.
* Extraction tuning: `--extraction {openrouter,fake,http} --extraction-model X --processing-batch-size 100 --processing-lease-seconds 180`.

## How to reproduce

```powershell
# End-to-end run (all 500 questions, OpenRouter extraction + embeddings)
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --extraction openrouter --workers 8 --token-budget 1500

# Judged subset (requires OPENROUTER_API_KEY in .env)
python benchmarks/longmemeval/run_benchmark.py --mode judged --task knowledge-update --limit 20 --workers 4 --budget-usd 8

# Internal ablation
python benchmarks/longmemeval/run_benchmark.py --mode retrieval-only --workers 8 --no-dense

# Per-category tables
foreach ($task in @("single-session-user","single-session-assistant","single-session-preference","knowledge-update","temporal-reasoning","multi-session")) {
  python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --task $task --limit 20 --workers 4 --extraction openrouter
}
```

Outputs are written to `results/longmemeval_s_{mode}_{timestamp}.json` with
`dataset.sha256`, `config`, `summary` (per-category rows), and per-question
`traces` (latency, packed token count, accepted/rejected counts, and — in judged
mode — `hypothesis` / `judge_verdict` / `correct`).

## Internal Retrieval Ablation (2026-08-25)

Retrieval run `longmemeval_s_retrieval_20260825-181122.json`, 500 questions,
`workers=8`, `--no-dense`, rerank enabled, `token-budget=1500`.

| Category | n | Recall@5 (%) | Recall@10 (%) | Recall@15 (%) | MRR@15 | NDCG@15 | Avg Context Tokens | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 70 | 98.6 | 98.6 | 100.0 | 0.98 | 0.984 | 1253.0 | 10678.1 |
| single-session-assistant | 56 | 100.0 | 100.0 | 100.0 | 1.0 | 1.0 | 1382.1 | 14999.1 |
| single-session-preference | 30 | 73.3 | 83.3 | 93.3 | 0.586 | 0.666 | 661.2 | 10558.6 |
| knowledge-update | 78 | 100.0 | 100.0 | 100.0 | 0.994 | 0.991 | 1350.6 | 12459.7 |
| temporal-reasoning | 133 | 92.5 | 94.0 | 96.2 | 0.882 | 0.862 | 657.2 | 11646.9 |
| multi-session | 133 | 97.0 | 98.5 | 99.2 | 0.909 | 0.872 | 913.5 | 11568.0 |
| **Overall** | **500** | **95.4** | **96.8** | **98.4** | **0.916** | **0.906** | **998.4** | **11927.2** |

Comparison head-to-head against Supermemory's published LongMemEval-S numbers
(same dataset split, same metric, Recall@15 with aggregation). This is an
internal retrieval ablation, not the headline product claim:

| Category | TermyteDB (FTS+Rerank) | Supermemory | Zep | Full Context |
|---|---:|---:|---:|---:|
| SSU | **100.0** | 97.0 | 92.9 | 81.4 |
| SSA | **100.0** | 100.0 | 80.4 | 94.6 |
| SSP | **93.3** | 90.0 | 56.7 | 20.0 |
| KU | **100.0** | 99.0 | 83.3 | 78.2 |
| TR | **96.2** | 91.0 | 62.4 | 45.1 |
| MS | **99.2** | 93.0 | 57.9 | 44.3 |
| **Overall** | **98.4** | **95.0** | 71.2 | 60.2 |

Zep / Full Context figures from Supermemory's research page and the
[Hindsight paper](https://arxiv.org/abs/2512.12818). The comparison uses the
published table verbatim; TermyteDB was evaluated with the harness described
above.

### Judged accuracy (supplementary, gpt-4o-mini)

Retrieval is perfect on several hard categories, but answer generation with a
cheap model and no temporal conflict resolution still struggles when old and
new facts both appear in the packed context:

* single-session-user: **93.3%** (n=30, limit-ordered)
* knowledge-update: **45.0%** (n=20) — 100% recall, model picks stale value
* temporal-reasoning: **50.0%** (n=20)
* multi-session: **25.0%** (n=20) — multi-hop synthesis under token budget

This is an honest second layer, not the headline. The fix is a first-class
temporal filter (promote latest `timestamp`/`valid_from`, demote superseded
versions via `invalid_at`) plus a date-aware answer prompt that includes
`question_date`. It is tracked as follow-up work and is visible in
`results/longmemeval_s_judged_*.json`.

## Why this is credible for a resume

* Runs on the **official 500-question LongMemEval-S** with a recorded SHA256.
* Measures the **production memory path**, not a retrieval-only shortcut, for
  the headline claim.
* Uses OpenRouter extraction and OpenRouter-compatible embeddings in the
  product path.
* **Reproducible**: one command, dataset bundled, per-question isolated DBs,
  config + SHA echoed in every result file.
* **Honest about limits**: retrieval-only remains available as an ablation, but
  it is not the claim that matters.

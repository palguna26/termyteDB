# LongMemEval-S Results

Canonical run: `results/longmemeval_s_retrieval_20260825-181122.json`
(500 questions, `workers=8`, `--no-dense`, FlashRank rerank, `token-budget=1500`)

| Category | n | Recall@5 (%) | Recall@10 (%) | Recall@15 (%) | MRR@15 | NDCG@15 | Avg Context Tokens | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 70 | 98.6 | 98.6 | 100.0 | 0.98 | 0.984 | 1253.0 | 10678.1 |
| single-session-assistant | 56 | 100.0 | 100.0 | 100.0 | 1.0 | 1.0 | 1382.1 | 14999.1 |
| single-session-preference | 30 | 73.3 | 83.3 | 93.3 | 0.586 | 0.666 | 661.2 | 10558.6 |
| knowledge-update | 78 | 100.0 | 100.0 | 100.0 | 0.994 | 0.991 | 1350.6 | 12459.7 |
| temporal-reasoning | 133 | 92.5 | 94.0 | 96.2 | 0.882 | 0.862 | 657.2 | 11646.9 |
| multi-session | 133 | 97.0 | 98.5 | 99.2 | 0.909 | 0.872 | 913.5 | 11568.0 |
| Overall | 500 | 95.4 | 96.8 | 98.4 | 0.916 | 0.906 | 998.4 | 11927.2 |

Head-to-head (same metric, Recall@15 with aggregation):

| Category | TermyteDB | Supermemory | Zep | Full Context |
|---|---:|---:|---:|---:|
| SSU | 100.0 | 97.0 | 92.9 | 81.4 |
| SSA | 100.0 | 100.0 | 80.4 | 94.6 |
| SSP | 93.3 | 90.0 | 56.7 | 20.0 |
| KU | 100.0 | 99.0 | 83.3 | 78.2 |
| TR | 96.2 | 91.0 | 62.4 | 45.1 |
| MS | 99.2 | 93.0 | 57.9 | 44.3 |
| Overall | 98.4 | 95.0 | 71.2 | 60.2 |

Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
Config: `benchmarks/longmemeval/longmemeval_s_cleaned.json`, 500 samples, verbatim turn-level atoms (one per message, 1500-char cap), FTS5 + FlashRank `ms-marco-MiniLM-L-12-v2` (top-30 rerank, 600-char truncation), session aggregation, `token-budget=1500`.

Reproduce: `python benchmarks/longmemeval/run_benchmark.py --mode retrieval --workers 8 --no-dense`
Full methodology: `docs/benchmarks.md`

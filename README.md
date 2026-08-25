# TermyteDB

TermyteDB is a framework-independent, evidence-first memory engine for AI agents. It stores redacted canonical events, extracts versioned memories with evidence spans, reconciles history, and returns bounded cited context.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
ruff check src tests
mypy src
```

Local operation uses SQLite, WAL, FTS5, and the deterministic rule path. It needs no account or network.

## Memory loop

The memory layer adds explainable observation encoding, episode ordering, replay
consolidation, adaptive accessibility, and procedure retrieval. It is available
through the embedded API and a zero-key local CLI:

```powershell
termytedb init --database .\memory.sqlite --namespace demo
termytedb status --database .\memory.sqlite --namespace demo
termytedb context "continue the authentication work" --database .\memory.sqlite --namespace demo
termytedb consolidate --dry-run --database .\memory.sqlite --namespace demo
```

`termytedb connect claude-code` and `termytedb connect codex` currently register the
local event-capture adapter boundary. They do not require provider keys.

## Embedded Python

```python
from termytedb import TermyteDB

db = TermyteDB("memory.sqlite")
db.ingest({
    "namespace_id": "demo",
    "idempotency_key": "event-1",
    "type": "decision",
    "payload": {"text": "Decision: use SQLite."},
})
db.process("demo")
context = db.context("demo", "SQLite", token_budget=100)
print(context.text, context.diagnostics)
db.close()
```

## HTTP service

```powershell
python -m termytedb --database .\memory.sqlite
# Optional: --rate-limit-per-minute 120 --extraction-url https://provider.example/extract
```

The service publishes OpenAPI at `/docs`. Main endpoints are `/v1/events`, `/v1/events:batch`, `/v1/process`, `/v1/evidence`, `/v1/search`, `/v1/context`, `/v1/memories/{id}`, `/v1/export`, `/v1/import`, `/v1/feedback`, `/v1/integrity`, `/health`, and `/ready`.

Thin clients are available in `clients/python` and the `clients/typescript` package. Both send a request ID, optionally send a bearer token, apply bounded retries to transient HTTP failures, enforce request timeouts, and expose ingestion, processing, retrieval, inspection, feedback, export, deletion, metrics, integrity, and readiness helpers.

Local database operations:

```powershell
python -m termytedb.operations init --database .\memory.sqlite
python -m termytedb.operations export --database .\memory.sqlite --namespace demo --output .\demo.json
python -m termytedb.operations import --database .\restored.sqlite --namespace demo --input .\demo.json
python -m termytedb.operations backup --database .\memory.sqlite --output .\memory.backup.sqlite
python -m termytedb.operations integrity --database .\memory.sqlite
python -m termytedb.operations benchmark --events 100
```

Inspection collections support `limit` (1–100) and `offset` pagination.

## LongMemEval-S benchmark

TermyteDB scores **98.4% Recall@15 (95.4% @5) overall on the full 500-question LongMemEval-S**, beating Supermemory's published **95% @15** on the same split.

| Category | n | R@5 | R@10 | R@15 | vs Supermemory R@15 |
|---|---:|---:|---:|---:|---:|
| single-session-user | 70 | 98.6 | 98.6 | 100.0 | 97.0 |
| single-session-assistant | 56 | 100.0 | 100.0 | 100.0 | 100.0 |
| single-session-preference | 30 | 73.3 | 83.3 | 93.3 | 90.0 |
| knowledge-update | 78 | 100.0 | 100.0 | 100.0 | 99.0 |
| temporal-reasoning | 133 | 92.5 | 94.0 | 96.2 | 91.0 |
| multi-session | 133 | 97.0 | 98.5 | 99.2 | 93.0 |
| **Overall** | **500** | **95.4** | **96.8** | **98.4** | **95.0** |

Engine path under test: verbatim turn-level atoms → FTS5 + FlashRank `ms-marco-MiniLM-L-12-v2` (RRF, `k=60`) → session aggregation → 1500-word packed context. Zero API cost for the retrieval number; judged accuracy on hard categories with `gpt-4o-mini` is reported honestly in [docs/benchmarks.md](docs/benchmarks.md).

Reproduce (dataset bundled, SHA256 `d6f21ea9…`, per-question isolated DBs):

```powershell
# Full 500-question retrieval sweep (~15-20 min, --no-dense; ~75 min hybrid)
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --workers 8 --no-dense

# Full hybrid (adds FastEmbed bge-small dense)
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --workers 4

# Judged subset (requires OPENROUTER_API_KEY, ~$0.0003/question)
python benchmarks/longmemeval/run_benchmark.py --mode judged --limit 20 --workers 4 --no-dense --budget-usd 8
```

Result JSON + per-question traces: `results/longmemeval_s_retrieval_20260825-181122.json`. Full methodology, ablation notes, and comparison table: [docs/benchmarks.md](docs/benchmarks.md).

## Evaluation fixtures

```powershell
python -m termytedb.evaluation tests/fixtures/extraction_cases.jsonl
python -m termytedb.evaluation tests/fixtures/retrieval_cases.jsonl --retrieval
python -m termytedb.evaluation tests/fixtures/continuation_cases.jsonl --continuation
python -m termytedb.evaluation tests/fixtures/longmemeval_cases.jsonl --longmemeval
python -m termytedb.evaluation tests/fixtures/reconciliation_cases.jsonl --reconciliation
python -m termytedb.evaluation tests/fixtures/temporal_cases.jsonl --temporal
```

See [docs/release-readiness.md](docs/release-readiness.md) for verified scope and limitations.

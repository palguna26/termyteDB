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

Thin clients are available in `clients/python` and the `clients/typescript` package. Both send a request ID, optionally send a bearer token, apply bounded retries to transient HTTP failures, enforce request timeouts, and expose ingest, processing, search, context, paginated inspection, memory history, and invalidation helpers.

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

## Evaluation and benchmarks

```powershell
python -m termytedb.evaluation tests/fixtures/extraction_cases.jsonl
python -m termytedb.evaluation tests/fixtures/retrieval_cases.jsonl --retrieval
python -m termytedb.evaluation tests/fixtures/continuation_cases.jsonl --continuation
python -m termytedb.evaluation tests/fixtures/longmemeval_cases.jsonl --longmemeval
python -m termytedb.evaluation tests/fixtures/reconciliation_cases.jsonl --reconciliation
python -m termytedb.evaluation tests/fixtures/temporal_cases.jsonl --temporal
```

See [docs/release-readiness.md](docs/release-readiness.md) for verified scope and limitations.

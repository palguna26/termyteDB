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
```

The service publishes OpenAPI at `/docs`. Main endpoints are `/v1/events`, `/v1/events:batch`, `/v1/process`, `/v1/search`, `/v1/context`, `/v1/memories/{id}`, `/v1/export`, `/v1/import`, `/v1/feedback`, `/v1/integrity`, `/health`, and `/ready`.

Inspection collections support `limit` (1–100) and `offset` pagination.

## Evaluation and benchmarks

```powershell
python -m termytedb.evaluation tests/fixtures/extraction_cases.jsonl
python -m termytedb.evaluation tests/fixtures/retrieval_cases.jsonl --retrieval
python -m termytedb.evaluation tests/fixtures/continuation_cases.jsonl --continuation
python -m termytedb.evaluation tests/fixtures/longmemeval_cases.jsonl --longmemeval
python -m termytedb.evaluation tests/fixtures/reconciliation_cases.jsonl --reconciliation
```

See [docs/release-readiness.md](docs/release-readiness.md) for verified scope and limitations.

# TermyteDB

TermyteDB is an embedded memory engine for AI agents. It stores conversation events in SQLite, extracts durable memories with evidence, reconciles updates, and retrieves bounded context.

## Install

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Use

```python
from src import TermyteDB

db = TermyteDB("memory.sqlite")
db.ingest({
    "namespace_id": "demo",
    "idempotency_key": "event-1",
    "type": "decision",
    "payload": {"text": "Decision: use SQLite."},
})
db.process("demo")
print(db.context("demo", "database choice").text)
db.close()
```

## LongMemEval-S

The dataset and single benchmark runner live in `benchmarks/longmemeval`.

```powershell
python -m pip install -e ".[benchmark]"
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --confirm-benchmark
```

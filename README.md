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
print(db.context("demo", "database choice").text)
db.close()
```

## LongMemEval-S

The dataset and single benchmark runner live in `benchmarks/longmemeval`.

```powershell
python -m pip install -e ".[benchmark]"
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --confirm-benchmark
```

### LongMemEval-Micro (30 samples, ~94% cheaper)

Stratified subset of LongMemEval-S with 5 questions per category (30 total) across
`single-session-user`, `single-session-assistant`, `single-session-preference`,
`knowledge-update`, `temporal-reasoning`, `multi-session`.

```powershell
# Retrieval-only (zero LLM cost) on micro subset
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --micro --confirm-benchmark

# End-to-end (OpenRouter extraction/embedding) on micro — ~16× cheaper than full 500
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --micro --confirm-benchmark

# Or point explicitly at the micro file
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --data-path benchmarks/longmemeval/longmemeval_micro.json --confirm-benchmark

# Regenerate the subset (deterministic seed 42, excludes _abs abstentions)
python benchmarks/longmemeval/create_micro.py
```

Files: `benchmarks/longmemeval/longmemeval_micro.json` and alias `longmemeval-micro.json`.

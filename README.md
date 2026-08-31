# TermyteDB

TermyteDB is an embedded memory engine for AI agents. It stores conversation events in SQLite, extracts durable memories with evidence, reconciles updates, and retrieves relevant memories via search. TermyteDB returns memories — the caller decides how those memories become model context.

## Install

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Use

Provider is explicit. Use `FakeExtractionProvider` for offline/tests and `OpenRouterExtractionProvider` in production.

```python
from src import TermyteDB
from src.memory.provider import FakeExtractionProvider

# Offline / tests (no network, single LLM call per ingest by default)
db = TermyteDB("memory.sqlite", extraction_provider=FakeExtractionProvider())
db.ingest({
    "namespace_id": "demo",
    "idempotency_key": "event-1",
    "type": "decision",
    "payload": {"text": "Decision: use SQLite."},
})
results = db.search("demo", "database choice", limit=5)
for memory in results:
    print(memory.statement)
db.close()
```

Production:

```python
import os
from src import TermyteDB
from src.memory.provider import OpenRouterExtractionProvider

db = TermyteDB(
    "memory.sqlite",
    extraction_provider=OpenRouterExtractionProvider(
        model=os.environ["TERMYTEDB_EXTRACTION_MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
)
# Extraction uses one Mem0-style LLM call per batch and returns a small
# {"memory": ["..."]} list. Optional LLM reconciliation is off by default.
# os.environ["TERMYTEDB_RECONCILIATION_ENABLED"] = "1"
db.ingest({
    "namespace_id": "demo",
    "idempotency_key": "event-1",
    "type": "decision",
    "payload": {"text": "Decision: use SQLite."},
})
# If ingest raises ProviderError (e.g. 429), the event is durably stored and a retryable
# processing job remains. Retry with:
#   db.process("demo")
results = db.search("demo", "database choice", limit=5)
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

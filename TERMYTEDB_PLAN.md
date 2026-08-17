# TermyteDB plan

## Product definition

TermyteDB is an independent, framework-neutral memory engine for agent evidence: conversations, tool calls, results, decisions, failures, corrections, and documents become versioned, evidence-backed memories. It isolates organization/project/user/agent scopes, retrieves bounded task context with provenance, supports local and hosted deployment, and is evaluated on coding-agent continuation and LongMemEval-s.

## Differentiation

The first measurable edge is evidence correctness, temporal/contradiction lifecycle, storage-enforced isolation, abstention, inspectability, and coding-agent continuation—not provider count or graph complexity. Gates are in [`docs/architecture/evaluation.md`](docs/architecture/evaluation.md) and [`docs/research/07-gaps-and-opportunities.md`](docs/research/07-gaps-and-opportunities.md).

## Architecture and technology

Python engine and Pydantic schemas; HTTP service; SQLAlchemy/Alembic; SQLite + FTS5 for local authority; PostgreSQL + FTS/pgvector for hosted; optional filesystem/object artifacts; small model/embedder/tokenizer interfaces. Evidence and memory versions are authoritative; indexes are rebuildable. Details: [`docs/architecture/termytedb-v1.md`](docs/architecture/termytedb-v1.md), [`docs/architecture/storage.md`](docs/architecture/storage.md).

## V1 API

`ingest`, `process`, `search`, `context`, `get memory`, `invalidate memory`, and `submit feedback`. CLI, SDK, and framework adapters remain external clients. See [`docs/architecture/api.md`](docs/architecture/api.md).

## Package shape

```text
termytedb/
  domain/        models and lifecycle rules
  storage/       SQLite/Postgres providers
  processing/    jobs, extraction, validation, reconciliation
  retrieval/     FTS/vector candidates, scoring, context
  providers/     model/embedder/tokenizer interfaces
  service/       HTTP schemas and routes
  diagnostics/   traces, metrics, cost
benchmarks/      component, retrieval, continuation, LongMemEval-s
```

## Milestones

1. Event-to-cited-context vertical slice.
2. Local alpha with versions, contradictions, jobs, recovery, and isolation.
3. Benchmarkable engine with labelled gates and abstention.
4. Coding-agent continuation readiness.
5. Thin Python/TypeScript SDKs.
6. Hosted PostgreSQL/worker readiness.
7. Graph and organizational features only after experiments.

Full scope and exit conditions: [`docs/plan/build-plan.md`](docs/plan/build-plan.md).

## Explicit non-goals

No Codex/Claude/Cursor/Termyte CLI logic; no agent runtime; no graph database in V1; no speculative provider matrix; no leaderboard-only optimization; no hosted billing or multi-region replication before the core gates pass.

## Major risks and unresolved decisions

The largest risks are unsupported model claims, temporal/contradiction errors, scope leakage, job loss/duplication, and context waste. Experiments are listed in [`docs/plan/open-questions.md`](docs/plan/open-questions.md); mitigations are in [`docs/plan/risk-register.md`](docs/plan/risk-register.md).

## Immediate first 10 engineering tasks

1. Create the Python package skeleton and locked dependency file.
2. Define Pydantic schemas for namespace, event, evidence, memory, version, and context.
3. Implement SQLite migrations and scope-required repository methods.
4. Implement idempotent event ingest with content hashes.
5. Add redaction and deterministic payload normalization.
6. Implement a rule-based extractor for decisions/failures with evidence spans.
7. Implement candidate validation and append-only memory versions.
8. Add FTS5 indexing and explainable search/context assembly.
9. Add a local worker with leases, retries, and dead-letter records.
10. Build the first fixture and test event → memory → cited context after restart.

## Research index

- [`docs/research/00-evaluation-framework.md`](docs/research/00-evaluation-framework.md)
- [`docs/research/01-supermemory.md`](docs/research/01-supermemory.md)
- [`docs/research/02-cognee.md`](docs/research/02-cognee.md)
- [`docs/research/03-tencentdb-agent-memory.md`](docs/research/03-tencentdb-agent-memory.md)
- [`docs/research/04-mem0.md`](docs/research/04-mem0.md)
- [`docs/research/05-graphiti.md`](docs/research/05-graphiti.md)
- [`docs/research/06-comparison-matrix.md`](docs/research/06-comparison-matrix.md)
- [`docs/research/07-gaps-and-opportunities.md`](docs/research/07-gaps-and-opportunities.md)


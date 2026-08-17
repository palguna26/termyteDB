# TermyteDB V1 release-readiness report

## Verified in this repository

- SQLite WAL lifecycle, transactional migrations, foreign-key checks, restart recovery, leased jobs, retries, dead letters, and crash rollback.
- Redacted immutable events with namespace-scoped idempotency, `event-v1` identity fields, bounded payloads, and artifact descriptors.
- Deterministic episodes, evidence-span validation, rule extraction, fake provider, generic HTTP provider boundary, append-only versions, corrections, disputes, invalidation, and audit decisions.
- Namespace-filtered FTS5 plus injectable embeddings (deterministic local default), explainable hybrid scores, context token budgets, abstention, selection diagnostics, feedback, export/import, deletion, backup, and integrity repair.
- Versioned HTTP service with request IDs, optional namespace authorization, health/readiness, evidence/job inspection, and OpenAPI generation through FastAPI.
- Deterministic extraction, retrieval, continuation, LongMemEval-shaped, reconciliation, and local performance harnesses, including concurrent namespace smoke coverage.

## Measured local evidence

- 98 automated tests pass with Ruff and mypy.
- Retrieval smoke fixture: Recall@5 `1.0`, MRR `1.0`, NDCG@5 `1.0`.
- Synthetic continuation fixture: no-memory `0.0`, previous-summary `0.0`, raw history `1.0`, TermyteDB `1.0`.
- Synthetic LongMemEval-shaped fixture: accuracy `1.0` over 2 items.
- Ten-event local smoke run: batch ingest `781.15 events/s`, processing `43.455 ms`, search `0.768 ms`, context `0.332 ms`, restart search `0.725 ms`.
- Wheel build succeeds with `python -m build --wheel --no-isolation`; generated build artifacts are not committed.

These are regression and smoke measurements, not broad product-quality, capacity, or external benchmark claims.

## Not verified or not implemented

- PostgreSQL/pgvector storage behavior and hosted transactional migrations.
- Real LongMemEval-s dataset execution and real coding-agent trajectories with executed verification commands.
- Real agent completion, tests-passed, tool-call, token, cost, and repeated-mistake measurements.
- Large-scale concurrent load, disk-failure simulation, and representative storage-growth targets.
- TypeScript/Python network SDK packages, pagination for any future collection endpoints beyond the current bounded inspection paths, and hosted authentication integration beyond the callback boundary.
- Artifact byte storage; only content-addressed descriptors are persisted.

## Release position

The local deterministic engine is evidence-backed and suitable for continued alpha evaluation. It is not proven as a complete hosted V1 until the limitations above are resolved or explicitly moved to post-V1 with an agreed product decision.

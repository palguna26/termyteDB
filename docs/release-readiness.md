# TermyteDB V1 release-readiness report

## Verified in this repository

- SQLite WAL lifecycle, transactional migrations, foreign-key checks, restart recovery, leased jobs, heartbeats, retries, dead letters, and crash rollback.
- Import-side-effect regression confirms importing the package and service modules creates no database or local files.
- Redacted immutable events with namespace-scoped idempotency, `event-v1` identity fields, bounded payloads, and artifact descriptors.
- Adversarial storage scan confirms a redacted secret is absent from the SQLite database and any WAL/journal files created during ingestion and processing.
- Namespace deletion storage scan confirms deleted secrets are absent from the SQLite database and WAL/journal files after close.
- Deterministic episodes, evidence-span validation, rule extraction, fake provider, generic HTTP provider boundary, failed-provider run audits, append-only versions, corrections, disputes, invalidation, and audit decisions.
- Provider token counts, latency, and configurable estimated extraction cost are persisted and included in namespace metrics without assuming pricing.
- Namespace-filtered FTS5 plus injectable embeddings (deterministic local default), explainable hybrid scores, context token budgets, abstention, selection diagnostics, feedback, export/import, deletion, backup, and integrity repair.
- Versioned HTTP service with request IDs, optional namespace authorization, health/readiness, paginated event and dedicated evidence/memory/job inspection, namespace metrics, and OpenAPI generation with a contract test.
- Thin Python and TypeScript clients with bounded transient retries, request IDs, timeouts, structured errors, paginated inspection, memory history, and invalidation; Python behavior is contract-tested and the TypeScript source compiles with TypeScript 7.0.2.
- Deterministic extraction, retrieval, temporal, continuation, LongMemEval-shaped, reconciliation, and local performance harnesses, including concurrent namespace smoke coverage.

## Measured local evidence

- 123 automated tests pass with Ruff and mypy.
- Retrieval smoke fixture: Recall@5 `1.0`, Precision@5 `0.2`, MRR `1.0`, NDCG@5 `1.0`.
- Namespace isolation fixture: zero search leaks and zero context leaks.
- Synthetic continuation fixture: no-memory `0.0`, previous-summary `0.0`, raw history `1.0`, TermyteDB `1.0`.
- Synthetic LongMemEval-shaped fixture: accuracy `1.0` over 2 items.
- Ten-event local smoke run: batch ingest `781.15 events/s`, processing `43.455 ms`, search `0.768 ms`, context `0.332 ms`, restart search `0.725 ms`.
- Wheel build succeeds with `python -m build --wheel --no-isolation`; generated build artifacts are not committed.
- Isolated wheel install smoke passed: installed `TermyteDB` completed ingest, processing, and cited context retrieval with an explicit SQLite path.
- Local operations CLI covers init, export/import, backup, integrity, and benchmark workflows.

These are regression and smoke measurements, not broad product-quality, capacity, or external benchmark claims.

## Not verified or not implemented

- PostgreSQL/pgvector storage behavior and hosted transactional migrations.
- Real LongMemEval-s dataset execution and real coding-agent trajectories with executed verification commands.
- Real agent completion, tests-passed, tool-call, token, cost, and repeated-mistake measurements.
- Large-scale concurrent load, disk-failure simulation, and representative storage-growth targets.
- Pagination for any future collection endpoints beyond the current bounded inspection paths, and hosted authentication integration beyond the callback boundary.
- Artifact byte storage; only content-addressed descriptors are persisted.

## Release position

The local deterministic engine is evidence-backed and suitable for continued alpha evaluation. It is not proven as a complete hosted V1 until the limitations above are resolved or explicitly moved to post-V1 with an agreed product decision.

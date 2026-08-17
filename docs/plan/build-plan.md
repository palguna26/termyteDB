# Founder-executable build plan

| Milestone | Deliverables and tests | Exit gate | Effort/risk |
|---|---|---|---|
| 1. Vertical prototype | SQLite schema, ingest, immutable event, one rule extractor, FTS search, context response, CLI-free HTTP test | Event to cited context works after restart; duplicate ingest is one event | 3-5 days; schema churn |
| 2. Local alpha | Pydantic API, redaction, episodes, memory versions, evidence validator, invalidation, job worker, diagnostics | 100% memories cite evidence; kill/restart recovery; isolation tests pass | 1-2 weeks; lifecycle bugs |
| 3. Benchmarkable engine | labelled component fixtures, vector provider, explainable hybrid rank, abstention, feedback, benchmark runner | component and retrieval gates in `architecture/evaluation.md` | 1-2 weeks; labels/model variance |
| 4. Continuation readiness | repository trajectory fixture format, no-memory/raw-summary/vector baselines, TermyteDB context adapter | repeatable Agent B comparison with verification tests | 1-2 weeks; noisy agent outcomes |
| 5. SDK readiness | thin Python and TypeScript clients generated from stable schemas; retries and request IDs | clients pass contract tests against local service | 3-5 days; API freeze |
| 6. Hosted readiness | PostgreSQL/Alembic, pgvector optional, artifact store, auth middleware, worker lease backend, metrics | migration/replay, tenant leakage, load and p95 gates | 2-4 weeks; operational cost |
| 7. Post-MVP | graph-table experiment, entity resolution, hosted queue, organization policy UI | only ship features with ablation/ops evidence | variable; scope risk |

Milestone 1 exclusions: model extraction, graph DB, CLI integration, hosted auth, multi-tenant billing, and full SDKs. Every milestone must keep raw evidence replayable and must not import Termyte CLI code.


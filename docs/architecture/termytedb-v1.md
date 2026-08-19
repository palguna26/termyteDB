# TermyteDB V1 architecture

TermyteDB is a framework-independent memory engine. It accepts evidence, stores it immutably, derives memories, and returns bounded context with provenance. CLI, SDK, and framework adapters call the service; they are not engine modules.

```text
CLI / SDK / adapters
        |
HTTP service or in-process engine
        |
scope + policy | evidence | processing | reconciliation | retrieval | diagnostics
        |
SQLite MVP (authoritative rows, FTS, optional vectors)
```

V1 uses Python, Pydantic, SQLAlchemy/Alembic, SQLite FTS5, and a provider interface for embeddings/models. Hosted readiness targets PostgreSQL and pgvector without changing the domain API. Object storage and an external queue are post-MVP.

The authoritative record is immutable evidence plus memory versions and evidence references. FTS/vector indexes are rebuildable. Graph tables are optional relationship indexes, not the source of truth.

## Boundaries

- Engine: domain models, transactions, processors, retrieval, lifecycle, providers.
- HTTP service: authentication, request validation, serialization, health, limits.
- Storage providers: SQLite now; PostgreSQL later; no provider leaks into domain code.
- Model providers: extraction and required embeddings.
- CLI/SDK/adapters: event mapping and client ergonomics, in separate repositories/packages.
- Benchmark harness: feeds the same public ingest/process/search/context API.

## V1 scope

Evidence events, artifacts, episodes, typed memories, versions, evidence links, namespaces, lexical retrieval, optional vector retrieval, temporal validity, contradiction/supersession, feedback, jobs, diagnostics, deletion, and local service. Exclude graph databases, autonomous agents, provider-specific adapters, hosted billing, and multi-region replication.

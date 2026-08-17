# ADR 0017: Production-path evaluation fixtures

## Status

Accepted

## Decision

Evaluation fixtures invoke the same public embedded engine used by local applications: ingest, processing, and retrieval. Metrics are calculated from labelled expected statements and include Recall@k, MRR, and NDCG@k. Temporary SQLite storage is used for isolation and is removed after each run.

## Consequences

The smoke result is reproducible and detects regressions in the complete local path. It does not represent a broad benchmark or compare external systems; larger labelled datasets and continuation cases remain required before release claims.

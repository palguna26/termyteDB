# ADR 0019: Explainable local hybrid retrieval

## Status

Accepted

## Decision

SQLite FTS5 remains the lexical source and a deterministic 32-dimensional hash embedder provides an offline vector source. Embeddings are rebuildable rows keyed by namespace and memory version. Retrieval applies namespace, current-status, and validity predicates in SQL, then combines normalized lexical and vector scores with weights 0.6 and 0.4. Vector-only candidates require a conservative 0.75 similarity threshold so unrelated memories are not returned for a lexical query.

Search responses expose both component scores. The local embedder is a deterministic baseline, not a trained semantic-quality claim; configurable external embedding providers remain a later upgrade.

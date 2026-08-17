# ADR 0029: Derived FTS and embedding integrity

## Status

Accepted

## Decision

Integrity checks treat both FTS and local embedding rows as rebuildable indexes. They report orphan and missing rows for each index. The existing explicit `repair_fts` command now rebuilds both indexes from active authoritative memory versions using the deterministic local embedder; it never guesses or changes evidence or memory truth.

## Consequences

Hybrid retrieval can detect and recover from missing or stale vector rows after corruption, import, or interrupted maintenance. The command name remains compatible with existing local scripts even though its repair scope is now both derived indexes.

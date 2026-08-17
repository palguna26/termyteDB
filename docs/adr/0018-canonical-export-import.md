# ADR 0018: Canonical namespace export and import

## Status

Accepted

## Decision

Namespace export is a JSON-compatible document containing the namespace, immutable events, memories, all memory versions and evidence references, processing jobs, extraction audit rows, and episodes. Import replays rows in foreign-key order with original IDs and `INSERT OR IGNORE`, then rebuilds the FTS index from authoritative memory rows.

Every imported row must carry the requested namespace. Mixed-namespace documents fail before successful completion through the surrounding transaction. Replaying the same document is idempotent.

## Consequences

Local backups can preserve evidence and history without relying on derived indexes. Large artifacts and hosted object storage remain separate concerns; this V1 document contains the current SQLite-authoritative data only.

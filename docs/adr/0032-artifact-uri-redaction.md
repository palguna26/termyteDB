# ADR 0032: Artifact URI redaction

## Status

Accepted

## Decision

Artifact metadata values are passed through deterministic redaction. An artifact URI is persisted only when it is the exact content-addressed `cas://<sha256>` reference matching the descriptor hash; all other URI forms are stored as `[REDACTED]`. This prevents credentials and arbitrary secret-bearing paths from remaining in SQLite, exports, or inspection responses.

## Consequences

External artifact locations are not retained in local V1 unless they are represented by a content-addressed reference. A future artifact store can map safe content addresses to external locations outside the evidence record.

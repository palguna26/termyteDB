# ADR 0040: Integrity-aware readiness

## Decision

The readiness endpoint runs the local integrity check and returns HTTP 503
when storage, schema, evidence, or derived indexes are unhealthy. Health
remains a liveness signal and does not inspect storage.

## Reason

Liveness and readiness have different meanings. A running process must not be
advertised as ready to receive work when its authoritative store is corrupt or
its schema is incompatible.

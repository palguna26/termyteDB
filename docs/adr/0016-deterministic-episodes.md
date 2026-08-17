# ADR 0016: Deterministic episode boundaries

## Status

Accepted

## Decision

Local V1 creates an episode during event append. Events with the same stream and a timestamp within 30 minutes of an existing episode boundary join that episode; otherwise a new episode is created. A missing stream is treated as its own deterministic event sequence. Episode membership is stored in a namespace-scoped join table, with event order preserved by an ordinal and boundaries recalculated for late events.

Model-assisted boundary detection is deferred until deterministic boundaries are measured against labelled data. Episode construction is independent of extraction and does not alter immutable events.

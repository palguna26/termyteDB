# ADR 0051: Temporal evaluation runner

## Decision

Add a deterministic `--temporal` evaluator using labelled validity intervals.
It runs the production ingest, configured extraction provider, reconciliation,
and retrieval paths, then reports stale-memory rejection and historical-state
accuracy.

## Reason

Temporal correctness must be measured through the same path used by clients;
unit validation alone cannot prove that expired knowledge is excluded while
explicit history remains available.

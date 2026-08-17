# ADR 0053: Persisted memory importance

## Decision

Memories store an `importance` value in the range 0 to 1 through a transactional
schema migration. Extraction candidates may provide it, with `0.5` as the
deterministic default. Memory inspection and retrieval component scores expose
the stored value.

## Reason

Importance is a retrieval and audit signal distinct from confidence. Keeping it
persisted makes ranking explanations reproducible and allows future evaluators
to measure its value without changing the evidence contract.

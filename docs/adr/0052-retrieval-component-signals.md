# ADR 0052: Retrieval component signals

## Decision

Search results expose component signals for confidence, evidence quality,
memory-type match, temporal status, and staleness penalty in addition to the
lexical and vector scores. The initial lexical/vector ranking weights remain
unchanged.

## Reason

Operators and evaluators need to explain why a result was returned without
claiming that the signals are trained weights. Keeping them additive and
inspectable preserves the current deterministic ranking contract.

# ADR 0035: Expanded conservative rule memory kinds

## Status

Accepted

## Decision

The offline rule extractor recognizes explicit `Outcome:`, `Constraint:`, `Procedure:`, `Attempt:`, `Task:`, and `Question:` labels in addition to decisions, failures, and corrections. It preserves the declared kind and exact full-match evidence span. It does not infer unlabeled facts or task state from ordinary prose.

## Consequences

Rule-only local operation now covers the declared structured memory kinds without model calls or unsupported inference. Rich semantic extraction remains behind the untrusted provider boundary.

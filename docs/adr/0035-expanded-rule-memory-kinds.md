# ADR 0035: Expanded conservative rule memory kinds

## Status

Accepted

## Decision

The offline rule extractor recognizes explicit `Outcome:`, `Constraint:`, `Procedure:`, `Attempt:`, `Task:`, and `Question:` labels in addition to decisions, failures, and corrections. It also recognizes a narrow sentence-start declarative fact form such as `The service runs on SQLite.` It preserves the declared kind and exact full-match evidence span, while rejecting speculative mid-sentence prose.

## Consequences

Rule-only local operation now covers the declared structured memory kinds without model calls or unsupported inference. Rich semantic extraction remains behind the untrusted provider boundary.

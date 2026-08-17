# ADR 0020: Feedback and request correlation

## Status

Accepted

## Decision

Feedback is a namespace-scoped append-only table referencing a memory, with constrained labels (`useful`, `not_useful`, `wrong`, `stale`) and redacted notes. The service adds an `X-Request-ID` response header, reusing a valid caller value or generating one when absent.

## Consequences

Retrieval quality signals can be collected without modifying authoritative memory directly, and HTTP logs can be correlated across a request. Feedback does not automatically change truth or ranking until measured evaluation justifies that behavior.
